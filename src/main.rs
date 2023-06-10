use etvr_hsf::HSF;
use opencv::imgproc::LINE_8;
use opencv::videoio::VideoWriter;
use opencv::core::Size;
use opencv::prelude::*;

const VIDEO_PATH: &str = "./assets/demo3.mp4";

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let framerate = 60.0;
    let size = Size::new(200, 150);
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut out_vid = VideoWriter::new("./assets/demo3.out.mp4", fourcc, framerate, size, true)?;

    let mut hsf = HSF::new();
    hsf.open_video(VIDEO_PATH)?;
    while hsf.read_frame()? {
        let start_time = std::time::Instant::now();
        match hsf.single_run() {
            Ok((center_x, center_y, frame, radius)) => {
                let elapsed = start_time.elapsed().as_micros();
                // avg ~300 microsecs on my 11th gen mobile i7 in release mode
                println!("elapsed: {} microsec", elapsed);

                println!("center_x: {}, center_y: {}, radius: {}", center_x, center_y, radius);

                opencv::imgproc::circle(
                    frame,
                    opencv::core::Point {
                        x: center_x as i32,
                        y: center_y as i32,
                    },
                    radius as i32,
                    opencv::core::Scalar::new(255., 0., 0., 0.),
                    1,
                    LINE_8,
                    0,
                )?;
                out_vid.write(frame)?;
            }
            Err(e) => {
                out_vid.write(&hsf.current_image_gray)?;
                println!("Error: {}", e);
            }
        }
    }
    out_vid.release()?;
    Ok(())
}
