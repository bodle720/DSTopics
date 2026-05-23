# -*- coding: utf-8 -*-
"""
Generate a synthetic pendulum image sequence for DMD experiments.

Outputs:
- RGB frame images
- foreground masks including the arm and bob
- bob-only masks
- ground-truth CSV with angle and bob-center coordinates
- metadata JSON
- optional preview GIF
- optional README/display GIF

Default experiment output:
    DMD/outputs/synthetic_pendulum/

Default README/display GIF:
    DMD/docs/images/synthetic_pendulum_preview.gif

To run from the DMD/ folder:
python helpers/make_synthetic_pendulum.py --overwrite
"""

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class PendulumConfig:
    num_frames: int = 160
    fps: float = 30.0
    frame_width: int = 256
    frame_height: int = 256
    pivot_x: int = 128
    pivot_y: int = 40
    length_pixels: float = 120.0
    bob_radius: int = 12
    arm_width: int = 3
    amplitude_degrees: float = 26.0
    period_seconds: float = 2.0
    phase_degrees: float = 0.0
    damping_per_second: float = 0.0
    background_mode: str = "plain"
    background_color: tuple = (245, 245, 245)
    arm_color: tuple = (45, 45, 45)
    bob_color: tuple = (30, 80, 200)
    grid_spacing: int = 32
    noise_std: float = 0.0
    anti_alias_scale: int = 4
    random_seed: int = 42


def parse_pair(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format 'x,y'.")

    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Both values must be integers.") from exc


def parse_color(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected RGB format 'r,g,b'.")

    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be integers.") from exc

    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("RGB values must be between 0 and 255.")

    return rgb


def get_resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    frames_dir = output_dir / "frames"
    foreground_masks_dir = output_dir / "foreground_masks"
    bob_masks_dir = output_dir / "bob_masks"

    frames_dir.mkdir(parents=True, exist_ok=True)
    foreground_masks_dir.mkdir(parents=True, exist_ok=True)
    bob_masks_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "frames_dir": frames_dir,
        "foreground_masks_dir": foreground_masks_dir,
        "bob_masks_dir": bob_masks_dir,
    }


def make_background(config):
    height = config.frame_height
    width = config.frame_width
    base = np.full((height, width, 3), config.background_color, dtype=np.uint8)

    if config.background_mode == "plain":
        return base

    if config.background_mode == "grid":
        grid = base.copy()
        spacing = max(4, config.grid_spacing)
        grid[::spacing, :, :] = np.maximum(grid[::spacing, :, :] - 20, 0)
        grid[:, ::spacing, :] = np.maximum(grid[:, ::spacing, :] - 20, 0)
        return grid.astype(np.uint8)

    if config.background_mode == "gradient":
        y = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1, 1)
        x = np.linspace(0, 1, width, dtype=np.float32).reshape(1, -1, 1)
        gradient = 18 * y + 10 * x
        result = np.clip(base.astype(np.float32) - gradient, 0, 255)
        return result.astype(np.uint8)

    raise ValueError(f"Unknown background mode: {config.background_mode}")


def pendulum_state(frame_index, config):
    t = frame_index / config.fps
    amplitude = math.radians(config.amplitude_degrees)
    phase = math.radians(config.phase_degrees)
    omega = 2.0 * math.pi / config.period_seconds
    decay = math.exp(-config.damping_per_second * t)
    argument = omega * t + phase

    theta = amplitude * decay * math.cos(argument)
    theta_dot = amplitude * decay * (
        -config.damping_per_second * math.cos(argument)
        - omega * math.sin(argument)
    )

    bob_x = config.pivot_x + config.length_pixels * math.sin(theta)
    bob_y = config.pivot_y + config.length_pixels * math.cos(theta)

    return {
        "frame_index": frame_index,
        "time_seconds": t,
        "theta_radians": theta,
        "theta_degrees": math.degrees(theta),
        "theta_dot_radians_per_second": theta_dot,
        "bob_x": bob_x,
        "bob_y": bob_y,
        "pivot_x": config.pivot_x,
        "pivot_y": config.pivot_y,
    }


def draw_rgb_frame(config, bob_x, bob_y, rng):
    width = config.frame_width
    height = config.frame_height
    scale = max(1, config.anti_alias_scale)

    background = make_background(config)
    image = Image.fromarray(background, mode="RGB")

    if scale > 1:
        image = image.resize((width * scale, height * scale), resample=Image.BICUBIC)

    draw = ImageDraw.Draw(image)

    pivot = (config.pivot_x * scale, config.pivot_y * scale)
    bob = (bob_x * scale, bob_y * scale)
    radius = config.bob_radius * scale

    draw.line(
        [pivot, bob],
        fill=config.arm_color,
        width=max(1, config.arm_width * scale),
    )

    draw.ellipse(
        [
            bob[0] - radius,
            bob[1] - radius,
            bob[0] + radius,
            bob[1] + radius,
        ],
        fill=config.bob_color,
        outline=(15, 35, 90),
        width=max(1, scale),
    )

    if scale > 1:
        image = image.resize((width, height), resample=get_resample_filter())

    frame = np.asarray(image).astype(np.float32)

    if config.noise_std > 0:
        noise = rng.normal(loc=0.0, scale=config.noise_std, size=frame.shape)
        frame = frame + noise

    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def draw_masks(config, bob_x, bob_y):
    width = config.frame_width
    height = config.frame_height
    pivot = (int(round(config.pivot_x)), int(round(config.pivot_y)))
    bob = (int(round(bob_x)), int(round(bob_y)))
    radius = int(round(config.bob_radius))

    foreground_mask = Image.new("L", (width, height), 0)
    bob_mask = Image.new("L", (width, height), 0)

    foreground_draw = ImageDraw.Draw(foreground_mask)
    bob_draw = ImageDraw.Draw(bob_mask)

    foreground_draw.line(
        [pivot, bob],
        fill=255,
        width=max(1, config.arm_width),
    )

    bob_bbox = [
        bob[0] - radius,
        bob[1] - radius,
        bob[0] + radius,
        bob[1] + radius,
    ]

    foreground_draw.ellipse(bob_bbox, fill=255)
    bob_draw.ellipse(bob_bbox, fill=255)

    return foreground_mask, bob_mask


def write_ground_truth_csv(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resize_for_gif(image, max_width=None):
    if max_width is None or image.width <= max_width:
        return image

    scale = max_width / image.width
    new_size = (max_width, int(round(image.height * scale)))
    return image.resize(new_size, resample=get_resample_filter())


def save_preview_gif(frame_paths, output_path, fps, max_frames=160, max_width=None):
    if not frame_paths:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(frame_paths) > max_frames:
        step = math.ceil(len(frame_paths) / max_frames)
        preview_paths = frame_paths[::step]
    else:
        step = 1
        preview_paths = frame_paths

    images = []
    for path in preview_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize_for_gif(image, max_width=max_width)
            images.append(image.copy())

    duration_ms = max(1, int(round(1000 * step / fps)))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    for image in images:
        image.close()


def write_metadata(config, output_dir, preview_gif_path, readme_gif_path, ground_truth_path):
    fit_frame_count = int(round(config.num_frames * 0.7))

    metadata = {
        "description": "Synthetic pendulum sequence for DMD, tracking, and forecasting experiments.",
        "config": asdict(config),
        "outputs": {
            "frames_dir": "frames",
            "foreground_masks_dir": "foreground_masks",
            "bob_masks_dir": "bob_masks",
            "ground_truth_csv": ground_truth_path.name,
            "preview_gif": preview_gif_path.name if preview_gif_path is not None else None,
            "readme_preview_gif": str(readme_gif_path) if readme_gif_path is not None else None,
        },
        "suggested_split": {
            "fit_frames": [0, max(0, fit_frame_count - 1)],
            "forecast_frames": [fit_frame_count, config.num_frames - 1],
            "fit_frame_count": fit_frame_count,
            "forecast_frame_count": config.num_frames - fit_frame_count,
        },
        "notes": [
            "Pixel coordinates use image convention: x increases to the right, y increases downward.",
            "theta=0 corresponds to the pendulum pointing straight down.",
            "foreground masks include both the pendulum arm and bob.",
            "bob masks include only the circular pendulum bob.",
            "The README/display GIF is a smaller presentation artifact intended for repository documentation.",
        ],
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)


def generate_sequence(
    config,
    output_dir,
    overwrite=False,
    save_gif=True,
    readme_gif_path=None,
    save_readme_gif=True,
    readme_gif_max_frames=96,
    readme_gif_max_width=420,
):
    paths = prepare_output_dir(output_dir, overwrite=overwrite)
    rng = np.random.default_rng(config.random_seed)

    rows = []
    frame_paths = []

    for frame_index in range(config.num_frames):
        state = pendulum_state(frame_index, config)

        frame = draw_rgb_frame(config, state["bob_x"], state["bob_y"], rng)
        foreground_mask, bob_mask = draw_masks(config, state["bob_x"], state["bob_y"])

        frame_name = f"frame_{frame_index:04d}.png"
        foreground_mask_name = f"foreground_mask_{frame_index:04d}.png"
        bob_mask_name = f"bob_mask_{frame_index:04d}.png"

        frame_path = paths["frames_dir"] / frame_name
        foreground_mask_path = paths["foreground_masks_dir"] / foreground_mask_name
        bob_mask_path = paths["bob_masks_dir"] / bob_mask_name

        frame.save(frame_path)
        foreground_mask.save(foreground_mask_path)
        bob_mask.save(bob_mask_path)

        frame_paths.append(frame_path)

        rows.append({
            **state,
            "frame_path": f"frames/{frame_name}",
            "foreground_mask_path": f"foreground_masks/{foreground_mask_name}",
            "bob_mask_path": f"bob_masks/{bob_mask_name}",
        })

    ground_truth_path = paths["output_dir"] / "ground_truth.csv"
    write_ground_truth_csv(rows, ground_truth_path)

    preview_gif_path = None
    if save_gif:
        preview_gif_path = paths["output_dir"] / "pendulum_preview.gif"
        save_preview_gif(
            frame_paths=frame_paths,
            output_path=preview_gif_path,
            fps=config.fps,
            max_frames=config.num_frames,
            max_width=None,
        )

    final_readme_gif_path = None
    if save_readme_gif and readme_gif_path is not None:
        final_readme_gif_path = Path(readme_gif_path)
        save_preview_gif(
            frame_paths=frame_paths,
            output_path=final_readme_gif_path,
            fps=config.fps,
            max_frames=readme_gif_max_frames,
            max_width=readme_gif_max_width,
        )

    write_metadata(
        config=config,
        output_dir=paths["output_dir"],
        preview_gif_path=preview_gif_path,
        readme_gif_path=final_readme_gif_path,
        ground_truth_path=ground_truth_path,
    )

    return {
        "output_dir": paths["output_dir"],
        "num_frames": config.num_frames,
        "ground_truth_csv": ground_truth_path,
        "preview_gif": preview_gif_path,
        "readme_gif": final_readme_gif_path,
    }


def build_arg_parser():
    project_dir = Path(__file__).resolve().parent.parent
    default_output_dir = project_dir / "outputs" / "synthetic_pendulum"
    default_readme_gif_path = default_output_dir / "readme_friendly" / "synthetic_pendulum_preview.gif"

    parser = argparse.ArgumentParser(
        description="Generate a synthetic pendulum sequence for DMD experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")

    parser.add_argument("--readme-gif-path", type=Path, default=default_readme_gif_path)
    parser.add_argument("--skip-readme-gif", action="store_true")
    parser.add_argument("--readme-gif-max-frames", type=int, default=96)
    parser.add_argument("--readme-gif-max-width", type=int, default=420)

    parser.add_argument("--num-frames", type=int, default=PendulumConfig.num_frames)
    parser.add_argument("--fps", type=float, default=PendulumConfig.fps)
    parser.add_argument("--frame-width", type=int, default=PendulumConfig.frame_width)
    parser.add_argument("--frame-height", type=int, default=PendulumConfig.frame_height)
    parser.add_argument("--pivot", type=parse_pair, default=(PendulumConfig.pivot_x, PendulumConfig.pivot_y))
    parser.add_argument("--length-pixels", type=float, default=PendulumConfig.length_pixels)
    parser.add_argument("--bob-radius", type=int, default=PendulumConfig.bob_radius)
    parser.add_argument("--arm-width", type=int, default=PendulumConfig.arm_width)

    parser.add_argument("--amplitude-degrees", type=float, default=PendulumConfig.amplitude_degrees)
    parser.add_argument("--period-seconds", type=float, default=PendulumConfig.period_seconds)
    parser.add_argument("--phase-degrees", type=float, default=PendulumConfig.phase_degrees)
    parser.add_argument("--damping-per-second", type=float, default=PendulumConfig.damping_per_second)

    parser.add_argument(
        "--background-mode",
        choices=["plain", "grid", "gradient"],
        default=PendulumConfig.background_mode,
    )
    parser.add_argument("--background-color", type=parse_color, default=PendulumConfig.background_color)
    parser.add_argument("--arm-color", type=parse_color, default=PendulumConfig.arm_color)
    parser.add_argument("--bob-color", type=parse_color, default=PendulumConfig.bob_color)
    parser.add_argument("--grid-spacing", type=int, default=PendulumConfig.grid_spacing)
    parser.add_argument("--noise-std", type=float, default=PendulumConfig.noise_std)
    parser.add_argument("--anti-alias-scale", type=int, default=PendulumConfig.anti_alias_scale)
    parser.add_argument("--random-seed", type=int, default=PendulumConfig.random_seed)

    return parser


def config_from_args(args):
    pivot_x, pivot_y = args.pivot

    return PendulumConfig(
        num_frames=args.num_frames,
        fps=args.fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        pivot_x=pivot_x,
        pivot_y=pivot_y,
        length_pixels=args.length_pixels,
        bob_radius=args.bob_radius,
        arm_width=args.arm_width,
        amplitude_degrees=args.amplitude_degrees,
        period_seconds=args.period_seconds,
        phase_degrees=args.phase_degrees,
        damping_per_second=args.damping_per_second,
        background_mode=args.background_mode,
        background_color=args.background_color,
        arm_color=args.arm_color,
        bob_color=args.bob_color,
        grid_spacing=args.grid_spacing,
        noise_std=args.noise_std,
        anti_alias_scale=args.anti_alias_scale,
        random_seed=args.random_seed,
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = config_from_args(args)

    summary = generate_sequence(
        config=config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        save_gif=not args.skip_gif,
        readme_gif_path=args.readme_gif_path,
        save_readme_gif=not args.skip_readme_gif,
        readme_gif_max_frames=args.readme_gif_max_frames,
        readme_gif_max_width=args.readme_gif_max_width,
    )

    print("Synthetic pendulum generation complete.")
    print(f"Output directory:      {summary['output_dir']}")
    print(f"Frames generated:      {summary['num_frames']}")
    print(f"Ground-truth CSV:      {summary['ground_truth_csv']}")

    if summary["preview_gif"] is not None:
        print(f"Preview GIF:           {summary['preview_gif']}")

    if summary["readme_gif"] is not None:
        print(f"README/display GIF:    {summary['readme_gif']}")


if __name__ == "__main__":
    main()