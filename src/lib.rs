use std::ops::AddAssign;
use std::time::{Duration, Instant};

use nalgebra::{DMatrix, SVector};
use opencv::core::{Mat, Point, Rect, Scalar, Size, BORDER_CONSTANT, CV_32FC1};
use opencv::imgproc::{
    CHAIN_APPROX_NONE, MORPH_CLOSE, MORPH_DILATE, MORPH_OPEN, MORPH_RECT, RETR_EXTERNAL,
    THRESH_BINARY_INV, LINE_8,
};
use opencv::prelude::*;
use opencv::types::VectorOfVectorOfPoint;
use opencv::videoio::{VideoCapture, CAP_ANY};

const IMSHOW_ENABLE: bool = false;
const SKIP_AUTORADIUS: bool = false;
const SKIP_BLINK_DETECT: bool = false;

// CV param
const DEFAULT_RADIUS: i32 = 20;
const AUTO_RADIUS_RANGE: (i32, i32) = (DEFAULT_RADIUS - 18, DEFAULT_RADIUS + 15); // (10, 30)
const AUTO_RADIUS_STEP: usize = 1;
const BLINK_INIT_FRAMES: usize = 60 * 3; // 60fps*3sec,Number of blink statistical frames
                                         // step==(x,y)
const DEFAULT_STEP: (usize, usize) = (5, 5); // bigger the steps,lower the processing time! ofc acc also takes an impact

pub struct CvParameters {
    radius: i32,
    pad: i32,
    step: (usize, usize),
    hsf: HaarSurroundFeature,
}

impl CvParameters {
    pub fn new(radius: i32, step: (usize, usize)) -> Self {
        let pad = 2 * radius;
        let hsf = HaarSurroundFeature::new(radius as usize);
        Self {
            radius,
            pad,
            step,
            hsf,
        }
    }

    pub fn get_rpsh(&self) -> (i32, i32, (usize, usize), &HaarSurroundFeature) {
        (self.radius, self.pad, self.step, &self.hsf)
    }

    pub fn get_radius(&self) -> i32 {
        self.radius
    }

    pub fn set_radius(&mut self, now_radius: i32) {
        self.radius = now_radius;
        self.pad = 2 * now_radius;
        self.hsf = HaarSurroundFeature::new(now_radius as usize);
    }

    pub fn get_step(&self) -> (usize, usize) {
        self.step
    }

    pub fn set_step(&mut self, now_step: (usize, usize)) {
        self.step = now_step;
    }

    pub fn get_hsf(&self) -> &HaarSurroundFeature {
        &self.hsf
    }

    pub fn set_hsf(&mut self, now_radius: usize) {
        self.hsf = HaarSurroundFeature::new(now_radius);
    }
}

pub struct HaarSurroundFeature {
    val_in: f64,
    val_out: f64,
    r_in: usize,
    r_out: usize,
}

impl HaarSurroundFeature {
    pub fn new(r_inner: usize) -> Self {
        let r_outer = r_inner * 3;
        let r_inner2 = r_inner * r_inner;
        let count_inner = r_inner2;
        let count_outer = r_outer * r_outer - r_inner2;
        let val_inner = 1.0 / r_inner2 as f64;
        let val_outer = -val_inner * count_inner as f64 / count_outer as f64;
        Self {
            val_in: val_inner,
            val_out: val_outer,
            r_in: r_inner,
            r_out: r_outer,
        }
    }

    pub fn get_kernel(&self) -> Vec<Vec<f64>> {
        let mut kernel = vec![vec![self.val_out; 2 * self.r_out - 1]; 2 * self.r_out - 1];
        let start = self.r_out - self.r_in;
        let end = self.r_out + self.r_in - 1;
        for i in start..end {
            for j in start..end {
                kernel[i][j] = self.val_in;
            }
        }
        kernel
    }
}

pub struct AutoRadiusCalc {
    response_list: Vec<(i32, f64)>,
    radius_cand_list: Vec<i32>,
    adj_comp_flag: bool,
    radius_middle_index: Option<usize>,
    left_item: Option<(i32, f64)>,
    right_item: Option<(i32, f64)>,
    left_index: Option<usize>,
    right_index: Option<usize>,
}

impl AutoRadiusCalc {
    pub fn new() -> Self {
        AutoRadiusCalc {
            response_list: vec![],
            radius_cand_list: vec![],
            adj_comp_flag: false,
            radius_middle_index: None,
            left_item: None,
            right_item: None,
            left_index: None,
            right_index: None,
        }
    }

    pub fn get_radius(&mut self) -> i32 {
        let prev_res_len = self.response_list.len();
        // adjustment of radius
        if prev_res_len == 1 {
            // len==1==response_list==[default_radius]
            self.adj_comp_flag = false;
            return AUTO_RADIUS_RANGE.0;
        } else if prev_res_len == 2 {
            // len==2==response_list==[default_radius, auto_radius_range[0]]
            self.adj_comp_flag = false;
            return AUTO_RADIUS_RANGE.1;
        } else if prev_res_len == 3 {
            // len==3==response_list==[default_radius,auto_radius_range[0],auto_radius_range[1]]
            if self.response_list[1].1 < self.response_list[2].1 {
                self.left_item = Some(self.response_list[1]);
                self.right_item = Some(self.response_list[0]);
            } else {
                self.left_item = Some(self.response_list[0]);
                self.right_item = Some(self.response_list[2]);
            }
            self.radius_cand_list = (self.left_item.unwrap().0..=self.right_item.unwrap().0)
                .step_by(AUTO_RADIUS_STEP)
                .collect();
            self.left_index = Some(0);
            self.right_index = Some(self.radius_cand_list.len() - 1);
            self.radius_middle_index =
                Some((self.left_index.unwrap() + self.right_index.unwrap()) / 2);
            self.adj_comp_flag = false;
            return self.radius_cand_list[self.radius_middle_index.unwrap()];
        } else {
            if let (Some(left_index), Some(right_index), Some(radius_middle_index)) =
                (self.left_index, self.right_index, self.radius_middle_index)
            {
                if left_index <= right_index && left_index != radius_middle_index {
                    let left_item = self.left_item.unwrap();
                    let right_item = self.right_item.unwrap();
                    if (left_item.1 + self.response_list.last().unwrap().1)
                        < (right_item.1 + self.response_list.last().unwrap().1)
                    {
                        self.right_item = Some(self.response_list.last().unwrap().clone());
                        self.right_index = Some(radius_middle_index - 1);
                        self.radius_middle_index =
                            Some((left_index + self.right_index.unwrap()) / 2);
                        self.adj_comp_flag = false;
                        return self.radius_cand_list[self.radius_middle_index.unwrap()];
                    }
                    if (left_item.1 + self.response_list.last().unwrap().1)
                        > (right_item.1 + self.response_list.last().unwrap().1)
                    {
                        self.left_item = Some(self.response_list.last().unwrap().clone());
                        self.left_index = Some(radius_middle_index + 1);
                        self.radius_middle_index =
                            Some((self.left_index.unwrap() + right_index) / 2);
                        self.adj_comp_flag = false;
                        return self.radius_cand_list[self.radius_middle_index.unwrap()];
                    }
                }
            }
            self.adj_comp_flag = true;
            return self.radius_cand_list[self.radius_middle_index.unwrap()];
        }
    }

    pub fn get_radius_base(&mut self) -> i32 {
        let prev_res_len = self.response_list.len();
        // adjustment of radius
        if prev_res_len == 1 {
            // len==1==response_list==[default_radius]
            self.adj_comp_flag = false;
            return AUTO_RADIUS_RANGE.0;
        } else if prev_res_len == 2 {
            // len==2==response_list==[default_radius, auto_radius_range[0]]
            self.adj_comp_flag = false;
            return AUTO_RADIUS_RANGE.1;
        } else if prev_res_len == 3 {
            // len==3==response_list==[default_radius,auto_radius_range[0],auto_radius_range[1]]
            let sort_res = self
                .response_list
                .iter()
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .clone();
            // Extract the radius with the lowest response value
            if sort_res.0 == DEFAULT_RADIUS {
                // If the default value is best, change now_mode to init after setting radius to the default value.
                self.adj_comp_flag = true;
                return DEFAULT_RADIUS;
            } else if sort_res.0 == AUTO_RADIUS_RANGE.0 {
                self.radius_cand_list = ((AUTO_RADIUS_RANGE.0 + AUTO_RADIUS_STEP as i32)
                    ..DEFAULT_RADIUS)
                    .step_by(AUTO_RADIUS_STEP)
                    .skip(1)
                    .collect();
                self.adj_comp_flag = false;
                return self.radius_cand_list.pop().unwrap();
            } else {
                self.radius_cand_list = ((DEFAULT_RADIUS + AUTO_RADIUS_STEP as i32)
                    ..AUTO_RADIUS_RANGE.1)
                    .step_by(AUTO_RADIUS_STEP)
                    .skip(1)
                    .collect();
                self.adj_comp_flag = false;
                return self.radius_cand_list.pop().unwrap();
            }
        } else {
            // Try the contents of the radius_cand_list in order until the radius_cand_list runs out
            // Better make it a binary search.
            if self.radius_cand_list.is_empty() {
                let sort_res = self
                    .response_list
                    .iter()
                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();
                self.adj_comp_flag = true;
                return sort_res.0;
            } else {
                self.adj_comp_flag = false;
                return self.radius_cand_list.pop().unwrap();
            }
        }
    }

    pub fn add_response(&mut self, radius: i32, response: f64) {
        self.response_list.push((radius, response));
    }
}

pub struct BlinkDetector {
    response_list: Vec<f64>,
    response_max: Option<f64>,
    enable_detect_flg: bool,
    quartile_1: Option<f64>,
}

impl BlinkDetector {
    pub fn new() -> Self {
        BlinkDetector {
            response_list: vec![],
            response_max: None,
            enable_detect_flg: false,
            quartile_1: None,
        }
    }

    pub fn calc_thresh(&mut self) {
        let mut sorted_response_list = self.response_list.iter().cloned().collect::<Vec<f64>>();
        sorted_response_list.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quartile_1_index = (sorted_response_list.len() as f32 * 0.25) as usize;
        let quartile_3_index = (sorted_response_list.len() as f32 * 0.75) as usize;
        let quartile_1 = sorted_response_list[quartile_1_index];
        let quartile_3 = sorted_response_list[quartile_3_index];
        let iqr = quartile_3 - quartile_1;
        self.quartile_1 = Some(quartile_1);
        self.response_max = Some(quartile_3 + (iqr * 1.5));
        self.enable_detect_flg = true;
    }

    pub fn detect(&self, now_response: f64) -> bool {
        match self.response_max {
            Some(max) => now_response > max,
            None => false,
        }
    }

    pub fn add_response(&mut self, response: f64) {
        self.response_list.push(response);
    }

    pub fn response_len(&self) -> usize {
        self.response_list.len()
    }
}

type Hist<T> = SVector<T, 256>;

/// Calculates the histogram of the image.
/// This is equivalent to `cv2.calcHist([image], [0], None, [256], [0, 256])`.
/// The image must be a single-channel grayscale image (CV_8UC1).
#[inline]
fn hist_calc(image: &Mat, out: &mut Hist<u32>) -> Result<(), opencv::Error> {
    for value in image.data_typed::<u8>()? {
        out[*value as usize] += 1;
    }
    Ok(())
}

/// Normalizes the histogram using the L1 norm.
/// This is equivalent to `cv2.normalize(hist, output, alpha=alpha, norm_type=cv2.NORM_L1)`.
#[inline]
fn hist_normalize(hist: &Hist<u32>, alpha: f32) -> Hist<f32> {
    let mut norm = hist.cast::<f32>();
    // Don't need to abs() since input was unsigned.
    let norm_factor = alpha / norm.iter().sum::<f32>();
    norm *= norm_factor;
    norm
}

/// Calculates the cumulative sum of the histogram.
#[inline]
fn hist_cumsum<T: nalgebra::Scalar + AddAssign>(hist: &Hist<T>) -> Hist<T> {
    let mut cumsum: Hist<T> = hist.clone_owned();
    for i in 1..cumsum.len() {
        let val = cumsum[i - 1].clone();
        cumsum[i] += val;
    }
    cumsum
}

/// Calculates the percentile of the histogram.
fn hist_percentile(hist: &Hist<u32>, percentile: f32) -> usize {
    let total = hist.sum();
    let frequencies = hist_cumsum(hist);
    let position = percentile * total as f32;
    for (i, &freq) in frequencies.iter().enumerate() {
        if freq as f32 >= position {
            return i;
        }
    }
    frequencies.len() - 1
}

const DEFAULT_ANCHOR: Point = Point::new(-1, -1);

fn morphology_ex(
    src: &Mat,
    dst: &mut Mat,
    op: i32,
    kernel: &Mat,
    iterations: i32,
) -> Result<(), opencv::Error> {
    opencv::imgproc::morphology_ex(
        src,
        dst,
        op,
        kernel,
        DEFAULT_ANCHOR,
        iterations,
        BORDER_CONSTANT,
        opencv::imgproc::morphology_default_border_value()?,
    )
}

pub struct CenterCorrection {
    hist_thr: f32,
    quartile_1: f64,
    frame_shape: Size,
    frame_bin: Mat,
    frame_final: Mat,
    morph_kernel: Mat,
    morph_kernel2: Mat,
    hist: Hist<u32>,
    hist_norm: Hist<f32>,
}

impl CenterCorrection {
    pub fn new(frame_shape: Size, quartile_1: f64) -> Result<Self, opencv::Error> {
        let morph_kernel = Mat::default();
        opencv::imgproc::get_structuring_element(MORPH_RECT, Size::new(7, 7), DEFAULT_ANCHOR)?;
        Ok(Self {
            hist_thr: 4.0,
            quartile_1,
            frame_shape,
            frame_bin: Mat::default(),
            frame_final: Mat::default(),
            morph_kernel,
            morph_kernel2: Mat::ones(3, 3, CV_32FC1)?.to_mat()?,
            hist: SVector::from_element(0),
            hist_norm: SVector::from_element(0.0),
        })
    }
    pub fn correction(
        &mut self,
        gray_frame: &Mat,
        orig_x: i32,
        orig_y: i32,
    ) -> Result<(i32, i32), opencv::Error> {
        let center_x = orig_x;
        let center_y = orig_y;

        // cv2.calcHist([gray_frame], [0], None, [256], [0, 256], hist=self.hist)
        hist_calc(gray_frame, &mut self.hist)?;

        // cv2.normalize(self.hist, self.hist_norm, alpha=100.0, norm_type=cv2.NORM_L1)
        self.hist_norm = hist_normalize(&self.hist, 1.0);

        // hist_per = self.hist_norm.cumsum()
        let hist_per = hist_cumsum(&self.hist_norm);

        // hist_index_list = self.hist_index[hist_per >= self.hist_thr]
        let hist_index_list = hist_per
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= self.hist_thr)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // frame_thr = hist_index_list[0] if len(hist_index_list) else np.percentile(cv2.bitwise_or(255 - self.frame_mask, gray_frame), 4)
        let frame_thr = match hist_index_list.first() {
            Some(&v) => v,
            None => hist_percentile(&self.hist, 0.04),
        };

        // self.frame_bin = cv2.threshold(gray_frame, frame_thr, 1, cv2.THRESH_BINARY_INV)[1]
        opencv::imgproc::threshold(
            gray_frame,
            &mut self.frame_bin,
            frame_thr as f64,
            1.0,
            THRESH_BINARY_INV,
        )?;

        // cropped_x, cropped_y, cropped_w, cropped_h = cv2.boundingRect(self.frame_bin)
        let cropped = opencv::imgproc::bounding_rect(&self.frame_bin)?;

        // self.frame_final = cv2.morphologyEx(self.frame_final, cv2.MORPH_CLOSE, self.morph_kernel)
        let mut tmp = Mat::default();
        morphology_ex(
            &self.frame_final,
            &mut tmp,
            MORPH_CLOSE,
            &self.morph_kernel,
            1,
        )?;

        // self.frame_final = cv2.morphologyEx(self.frame_final, cv2.MORPH_OPEN, self.morph_kernel)
        morphology_ex(
            &tmp,
            &mut self.frame_final,
            MORPH_OPEN,
            &self.morph_kernel,
            1,
        )?;

        let (base_x, base_y) = if cropped.height == self.frame_shape.height
            && cropped.width == self.frame_shape.width
        {
            (center_x, center_y)
        } else {
            let base_x = cropped.x + cropped.width / 2;
            let base_y = cropped.y + cropped.height / 2;
            if *self.frame_final.at_2d::<u8>(base_y, base_x)? == 1 {
                (base_x, base_y)
            } else if *self.frame_final.at_2d::<u8>(center_y, center_x)? == 1 {
                (center_x, center_y)
            } else {
                morphology_ex(
                    &self.frame_final,
                    &mut tmp,
                    MORPH_DILATE,
                    &self.morph_kernel2,
                    3,
                )?;
                self.frame_final = tmp;
                (base_x, base_y)
            }
        };

        // contours, _ = cv2.findContours(self.frame_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        let mut contours = VectorOfVectorOfPoint::new();
        opencv::imgproc::find_contours(
            &self.frame_final,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_NONE,
            Point::default(),
        )?;

        // contours_box = [cv2.boundingRect(cnt) for cnt in contours]
        let mut contours_box = Vec::<Rect>::with_capacity(contours.len());
        for contour in contours {
            contours_box.push(opencv::imgproc::bounding_rect(&contour)?);
        }

        // contours_dist = np.array([abs(base_x - (cnt_x + cnt_w / 2)) + abs(base_y - (cnt_y + cnt_h / 2)) for cnt_x, cnt_y, cnt_w, cnt_h in contours_box])
        let mut contours_dist = Vec::<i32>::with_capacity(contours_box.len());
        for contour in contours_box.iter() {
            contours_dist.push(
                (base_x - (contour.x + contour.width / 2)).abs()
                    + (base_y - (contour.y + contour.height / 2)).abs(),
            );
        }

        let (x, y) = if contours_box.len() > 0 {
            let min_index = contours_dist
                .iter()
                .enumerate()
                .max_by_key(|&(_, v)| v)
                .unwrap()
                .0;
            let cropped2 = contours_box[min_index];
            (
                cropped2.x + cropped2.width / 2,
                cropped2.y + cropped2.height / 2,
            )
        } else {
            (center_x, center_y)
        };

        let i_y = (y - 5).max(0);
        let i_y2 = (y + 5).min(self.frame_shape.height);
        let i_x = (x - 5).max(0);
        let i_x2 = (x + 5).min(self.frame_shape.width);
        let view = gray_frame
            .row_range(&opencv::core::Range::new(i_y, i_y2)?)?
            .col_range(&opencv::core::Range::new(i_x, i_x2)?)?;
        let mut min = 0.0;
        let mut max = 255.0;
        opencv::core::min_max_loc(
            &view,
            Some(&mut min),
            Some(&mut max),
            None,
            None,
            &opencv::core::no_array(),
        )?;

        if min < self.quartile_1 {
            Ok((x, y))
        } else {
            Ok((orig_x, orig_y))
        }
    }
}

fn get_hsf_center(
    padding: usize,
    x_step: usize,
    y_step: usize,
    min_loc: (usize, usize),
) -> (usize, usize) {
    (
        padding + (x_step * min_loc.0) - padding,
        padding + (y_step * min_loc.1) - padding,
    )
}

/*
def safe_crop(img, x, y, x2, y2, keepsize=False):
    # The order of the arguments can be reconsidered.
    img_h, img_w = img.shape[:2]
    outimg = img[max(0, y) : min(img_h, y2), max(0, x) : min(img_w, x2)].copy()
    reqsize_x, reqsize_y = abs(x2 - x), abs(y2 - y)
    if keepsize and outimg.shape[:2] != (reqsize_y, reqsize_x):
        # If the size is different from the expected size (smaller by the amount that is out of range)
        outimg = cv2.resize(outimg, (reqsize_x, reqsize_y))
    return outimg
 */

fn safe_crop(img: &Mat, x: i32, y: i32, x2: i32, y2: i32) -> Result<Mat, opencv::Error> {
    let img_h = img.rows();
    let img_w = img.cols();
    let outimg = Mat::roi(
        img,
        Rect::new(x.max(0), y.max(0), (x2 - x).min(img_w), (y2 - y).min(img_h)),
    )?;
    Ok(outimg)
}

pub struct HSF {
    cvparam: CvParameters,
    cv_modeo: Vec<&'static str>,
    now_modeo: &'static str,
    auto_radius_calc: AutoRadiusCalc,
    blink_detector: BlinkDetector,
    center_q1: BlinkDetector,
    center_correct: Option<CenterCorrection>,
    cap: Option<VideoCapture>,
    timedict_to_gray: Vec<Duration>,
    timedict_int_img: Vec<Duration>,
    timedict_conv_int: Vec<Duration>,
    timedict_crop: Vec<Duration>,
    timedict_total_cv: Vec<Duration>,
    pub current_image_gray: Mat,
}

impl HSF {
    pub fn new() -> Self {
        Self {
            cvparam: CvParameters::new(DEFAULT_RADIUS, DEFAULT_STEP),
            cv_modeo: vec!["first_frame", "radius_adjust", "blink_adjust", "normal"],
            now_modeo: "first_frame",
            auto_radius_calc: AutoRadiusCalc::new(),
            blink_detector: BlinkDetector::new(),
            center_q1: BlinkDetector::new(),
            center_correct: None,
            cap: None,
            timedict_to_gray: Vec::new(),
            timedict_int_img: Vec::new(),
            timedict_conv_int: Vec::new(),
            timedict_crop: Vec::new(),
            timedict_total_cv: Vec::new(),
            current_image_gray: Mat::default(),
        }
    }
    pub fn open_video(&mut self, path: &str) -> Result<(), opencv::Error> {
        let cap = VideoCapture::from_file(path, CAP_ANY)?;
        if !cap.is_opened()? {
            return Err(opencv::Error::new(
                opencv::core::StsError,
                format!("Error opening video stream or file: {}", path),
            ));
        }
        self.cap = Some(cap);
        Ok(())
    }
    pub fn read_frame(&mut self) -> Result<bool, opencv::Error> {
        if let Some(cap) = &mut self.cap {
            // let mut frame = Mat::default();
            // let ret = cap.read(&mut frame)?;
            let ret = cap.read(&mut self.current_image_gray)?;
            if ret {
                // opencv::imgproc::cvt_color(
                //     &frame,
                //     &mut self.current_image_gray,
                //     opencv::imgproc::COLOR_BGR2GRAY,
                //     0,
                // )?;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(opencv::Error::new(
                opencv::core::StsError,
                "VideoCapture is not opened",
            ))
        }
    }
    pub fn single_run(&mut self) -> Result<(usize, usize, &mut Mat, i32), opencv::Error> {
        let frame = &mut self.current_image_gray;
        if self.now_modeo == self.cv_modeo[1] {
            self.cvparam.radius = self.auto_radius_calc.get_radius();
            if self.auto_radius_calc.adj_comp_flag {
                self.now_modeo = if !SKIP_BLINK_DETECT {
                    self.cv_modeo[2]
                } else {
                    self.cv_modeo[3]
                }
            }
        }

        let (radius, pad, step, hsf) = self.cvparam.get_rpsh();
        let cv_start_time = Instant::now();

        let mut gray_frame = Mat::default();

        opencv::imgproc::cvt_color(
            frame,
            &mut gray_frame,
            opencv::imgproc::COLOR_BGR2GRAY,
            0,
        )?;

        self.timedict_to_gray.push(cv_start_time.elapsed());

        let int_start_time = Instant::now();

        let mut frame_pad = Mat::default();
        opencv::core::copy_make_border(
            &gray_frame,
            &mut frame_pad,
            pad,
            pad,
            pad,
            pad,
            BORDER_CONSTANT,
            Scalar::default(),
        )?;

        let mut frame_int = Mat::default();
        opencv::imgproc::integral(&frame_pad, &mut frame_int, opencv::core::CV_64FC1)?;

        let frame_int = DMatrix::<f64>::from_row_slice(
            frame_int.rows() as usize,
            frame_int.cols() as usize,
            frame_int.data_typed::<f64>()?,
        );

        let frame_shape = gray_frame.size()?;
        let pad = pad as usize;
        let x_step = step.0;
        let y_step = step.1;
        let r_in = hsf.r_in;
        let r_out = hsf.r_out;

        let size = frame_pad.size()?;
        let row = size.height as usize;
        let col = size.width as usize;

        let y_steps_arr = (pad..(row - pad)).step_by(y_step).collect::<Vec<usize>>();
        let x_steps_arr = (pad..(col - pad)).step_by(x_step).collect::<Vec<usize>>();
        let len_sx = x_steps_arr.len();
        let len_sy = y_steps_arr.len();
        let len_syx = (len_sy, len_sx);
        let y_end = pad + (y_step * (len_sy - 1));
        let x_end = pad + (x_step * (len_sx - 1));

        let y_rin_m = ((pad - r_in)..(y_end - r_in + 1))
            .step_by(y_step)
            .collect::<Vec<usize>>();
        let y_rin_p = ((pad + r_in)..(y_end + r_in + 1))
            .step_by(y_step)
            .collect::<Vec<usize>>();
        let x_rin_m = ((pad - r_in)..(x_end - r_in + 1))
            .step_by(x_step)
            .collect::<Vec<usize>>();
        let x_rin_p = ((pad + r_in)..(x_end + r_in + 1))
            .step_by(x_step)
            .collect::<Vec<usize>>();

        // in_p00 = frame_int[y_rin_m, x_rin_m]
        // in_p11 = frame_int[y_rin_p, x_rin_p]
        // in_p01 = frame_int[y_rin_m, x_rin_p]
        // in_p10 = frame_int[y_rin_p, x_rin_m]

        let in_p00 = frame_int
            .select_rows(y_rin_m.iter())
            .select_columns(x_rin_m.iter());
        let in_p11 = frame_int
            .select_rows(y_rin_p.iter())
            .select_columns(x_rin_p.iter());
        let in_p01 = frame_int
            .select_rows(y_rin_m.iter())
            .select_columns(x_rin_p.iter());
        let in_p10 = frame_int
            .select_rows(y_rin_p.iter())
            .select_columns(x_rin_m.iter());

        // y_ro_m = np.maximum(y_steps_arr - r_out, 0)  # [:,np.newaxis]
        // x_ro_m = np.maximum(x_steps_arr - r_out, 0)  # [np.newaxis,:]
        // y_ro_p = np.minimum(row, y_steps_arr + r_out)  # [:,np.newaxis]
        // x_ro_p = np.minimum(col, x_steps_arr + r_out)  # [np.newaxis,:]

        let y_ro_m = y_steps_arr
            .iter()
            .map(|&x| x.saturating_sub(r_out))
            .collect::<Vec<usize>>();
        let x_ro_m = x_steps_arr
            .iter()
            .map(|&x| x.saturating_sub(r_out))
            .collect::<Vec<usize>>();
        let y_ro_p = y_steps_arr
            .iter()
            .map(|&x| (x + r_out).min(row))
            .collect::<Vec<usize>>();
        let x_ro_p = x_steps_arr
            .iter()
            .map(|&x| (x + r_out).min(col))
            .collect::<Vec<usize>>();

        self.timedict_int_img.push(int_start_time.elapsed());

        let conv_int_start_time = Instant::now();

        let inner_sum = in_p00 + in_p11 - in_p01 - in_p10;
        let out_p_temp = frame_int.select_rows(y_ro_m.iter());
        let out_p00 = out_p_temp.select_columns(x_ro_m.iter());
        let out_p01 = out_p_temp.select_columns(x_ro_p.iter());
        let out_p_temp = frame_int.select_rows(y_ro_p.iter());
        let out_p11 = out_p_temp.select_columns(x_ro_p.iter());
        let out_p10 = out_p_temp.select_columns(x_ro_m.iter());

        let outer_sum = out_p00 + out_p11 - out_p01 - out_p10 - &inner_sum;
        let response_list = inner_sum * hsf.val_in + outer_sum * hsf.val_out;

        let mut min_response = f64::MAX;
        let mut min_loc: (usize, usize) = (0, 0);

        for (y, row) in response_list.row_iter().enumerate() {
            for (x, val) in row.iter().enumerate() {
                if *val < min_response {
                    min_response = *val;
                    min_loc = (x, y);
                }
            }
        }

        let frame_conv_stride = response_list;

        let center_xy = get_hsf_center(pad, step.0, step.1, min_loc);

        self.timedict_conv_int.push(conv_int_start_time.elapsed());

        let crop_start_time = Instant::now();

        let (mut center_x, mut center_y) = center_xy;
        let mut upper_x = center_x as i32 + radius;
        let mut lower_x = center_x as i32 - radius;
        let mut upper_y = center_y as i32 + radius;
        let mut lower_y = center_y as i32 - radius;

        let mut cropped_image = safe_crop(&gray_frame, lower_x, lower_y, upper_x, upper_y)?;

        if self.now_modeo == self.cv_modeo[0] || self.now_modeo == self.cv_modeo[1] {
            // If mode is first_frame or radius_adjust, record current radius and response
            self.auto_radius_calc.add_response(radius, min_response);
        } else if self.now_modeo == self.cv_modeo[2] {
            // Statistics for blink detection
            if self.blink_detector.response_len() < BLINK_INIT_FRAMES {
                let mean = opencv::core::mean(&cropped_image, &opencv::core::no_array())?;
                self.blink_detector.add_response(mean[0]);

                upper_x = center_x as i32 + 20.max(radius); // self.center_correct.center_q1_radius
                lower_x = center_x as i32 - 20.max(radius); // self.center_correct.center_q1_radius
                upper_y = center_y as i32 + 20.max(radius); // self.center_correct.center_q1_radius
                lower_y = center_y as i32 - 20.max(radius); // self.center_correct.center_q1_radius

                let crop = safe_crop(&gray_frame, lower_x, lower_y, upper_x, upper_y)?;
                let mean = opencv::core::mean(&crop, &opencv::core::no_array())?;
                self.center_q1.add_response(mean[0]);
            } else {
                self.blink_detector.calc_thresh();
                self.center_q1.calc_thresh();
                self.now_modeo = self.cv_modeo[3];
            }
        } else {
            // if 0 in cropped_image.shape
            if cropped_image.size()?.width == 0 || cropped_image.size()?.height == 0 {
                println!("Something's wrong.");
            } else {
                let orig_x = center_x;
                let orig_y = center_y;
                if self.blink_detector.enable_detect_flg {
                    // If the average value of cropped_image is greater than response_max
                    // (i.e., if the cropimage is whitish
                    let mean = opencv::core::mean(&cropped_image, &opencv::core::no_array())?;
                    if self.blink_detector.detect(mean[0]) {
                        // blink
                    } else {
                        let center_correct = match self.center_correct.as_mut() {
                            None => {
                                self.center_correct = Some(CenterCorrection::new(
                                    frame_shape,
                                    self.center_q1.quartile_1.unwrap(),
                                )?);
                                self.center_correct.as_mut().unwrap()
                            }
                            Some(center_correct) => {
                                if center_correct.frame_shape != frame_shape {
                                    *center_correct = CenterCorrection::new(
                                        frame_shape,
                                        self.center_q1.quartile_1.unwrap(),
                                    )?;
                                    center_correct
                                } else {
                                    center_correct
                                }
                            }
                        };

                        let center_xy = center_correct.correction(
                            &gray_frame,
                            center_x as i32,
                            center_y as i32,
                        )?;
                        center_x = center_xy.0 as usize;
                        center_y = center_xy.1 as usize;
                        upper_x = center_x as i32 + radius;
                        lower_x = center_x as i32 - radius;
                        upper_y = center_y as i32 + radius;
                        lower_y = center_y as i32 - radius;

                        cropped_image = safe_crop(&gray_frame, lower_x, lower_y, upper_x, upper_y)?;
                    }
                }

                opencv::imgproc::circle(
                    frame,
                    opencv::core::Point {
                        x: orig_x as i32,
                        y: orig_y as i32,
                    },
                    6,
                    opencv::core::Scalar::new(0., 0., 255., 0.),
                    1,
                    LINE_8,
                    0,
                )?;
                opencv::imgproc::circle(
                    frame,
                    opencv::core::Point {
                        x: center_x as i32,
                        y: center_y as i32,
                    },
                    3,
                    opencv::core::Scalar::new(255., 0., 0., 0.),
                    1,
                    LINE_8,
                    0,
                )?;

            }
        }

        self.timedict_crop.push(crop_start_time.elapsed());
        self.timedict_total_cv.push(cv_start_time.elapsed());

        // if self.now_modeo == self.cv_modeo[0]:
        //     # Moving from first_frame to the next mode
        //     if skip_autoradius and skip_blink_detect:
        //         self.now_modeo = self.cv_modeo[3]
        //     elif skip_autoradius:
        //         self.now_modeo = self.cv_modeo[2]
        //     else:
        //         self.now_modeo = self.cv_modeo[1]

        if self.now_modeo == self.cv_modeo[0] {
            // Moving from first_frame to the next mode
            if SKIP_AUTORADIUS && SKIP_BLINK_DETECT {
                self.now_modeo = self.cv_modeo[3];
            } else if SKIP_AUTORADIUS {
                self.now_modeo = self.cv_modeo[2];
            } else {
                self.now_modeo = self.cv_modeo[1];
            }
        }

        Ok((center_x, center_y, frame, radius))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
