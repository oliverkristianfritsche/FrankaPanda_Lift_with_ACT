#!/usr/bin/env python3
# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""View saved demonstrations."""

import argparse
import h5py
import numpy as np
import cv2
from pathlib import Path


def parse_observation(obs: np.ndarray) -> dict:
    """Parse observation vector into named components."""
    return {
        "joint_pos": obs[0:9],
        "joint_vel": obs[9:18],
        "object_pos": obs[18:21],
        "target_pos": obs[21:24],  # Just XYZ
        "target_quat": obs[24:28],  # Quaternion
        "last_action": obs[28:36],
    }


def draw_text_with_background(img, text, pos, font_scale=2.0, color=(255, 255, 255), 
                               bg_color=(0, 0, 0), thickness=2):
    """Draw text with a background rectangle for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # Draw background rectangle
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return text_h + baseline + 4  # Return height used


def render_frame(image: np.ndarray, obs: np.ndarray, action: np.ndarray, 
                 reward: float, step: int, total_steps: int, 
                 cumulative_reward: float, scale: int = 4) -> np.ndarray:
    \"\"\"Render a frame with overlaid information.\"\"\"
    h, w = image.shape[:2]
    img = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Parse observation
    parsed = parse_observation(obs)
    
    # Starting position for text
    y_offset = 45
    x_left = 10
    x_right = img.shape[1] // 2 + 10
    line_height = 0
    
    # Left column - Episode info and positions
    line_height = draw_text_with_background(
        img, f"Step: {step+1}/{total_steps}", (x_left, y_offset))
    y_offset += line_height
    
    line_height = draw_text_with_background(
        img, f"Reward: {reward:.3f} | Cumulative: {cumulative_reward:.2f}", (x_left, y_offset),
        color=(0, 255, 0) if reward > 0.1 else (255, 255, 255))
    y_offset += line_height
    
    # Object position
    obj_pos = parsed["object_pos"]
    line_height = draw_text_with_background(
        img, f"Obj: [{obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f}]", (x_left, y_offset),
        color=(255, 200, 0))
    y_offset += line_height
    
    # Target position
    tgt_pos = parsed["target_pos"]
    line_height = draw_text_with_background(
        img, f"Tgt: [{tgt_pos[0]:.2f}, {tgt_pos[1]:.2f}, {tgt_pos[2]:.2f}]", (x_left, y_offset),
        color=(0, 255, 255))
    y_offset += line_height
    
    # Distance to goal
    dist = np.linalg.norm(obj_pos - tgt_pos)
    dist_color = (0, 255, 0) if dist < 0.05 else (255, 255, 0) if dist < 0.1 else (255, 255, 255)
    line_height = draw_text_with_background(
        img, f"Dist to goal: {dist:.3f}m", (x_left, y_offset), color=dist_color)
    y_offset += line_height + 5
    
    # Joint positions (compact format)
    jp = parsed["joint_pos"]
    line_height = draw_text_with_background(
        img, f"Joints: [{jp[0]:.1f},{jp[1]:.1f},{jp[2]:.1f},{jp[3]:.1f}]", (x_left, y_offset),
        color=(200, 200, 255))
    y_offset += line_height
    line_height = draw_text_with_background(
        img, f"       [{jp[4]:.1f},{jp[5]:.1f},{jp[6]:.1f},{jp[7]:.1f},{jp[8]:.1f}]", (x_left, y_offset),
        color=(200, 200, 255))
    y_offset += line_height + 5
    
    # Actions - all in left column
    act = action
    line_height = draw_text_with_background(
        img, "Actions:", (x_left, y_offset), color=(255, 150, 150))
    y_offset += line_height
    
    # Arm actions (7 joints)
    line_height = draw_text_with_background(
        img, f"Arm: [{act[0]:.2f},{act[1]:.2f},{act[2]:.2f},{act[3]:.2f}]", (x_left, y_offset),
        color=(255, 150, 150))
    y_offset += line_height
    line_height = draw_text_with_background(
        img, f"     [{act[4]:.2f},{act[5]:.2f},{act[6]:.2f}]", (x_left, y_offset),
        color=(255, 150, 150))
    y_offset += line_height
    
    # Gripper action (just the value)
    # line_height = draw_text_with_background(
    #     img, f"Gripper: {act[7]:.2f}", (x_left, y_offset),
    #     color=(0, 200, 255))
    
    # Progress bar at bottom
    bar_height = 8
    bar_y = img.shape[0] - bar_height - 5
    bar_width = img.shape[1] - 20
    progress = (step + 1) / total_steps
    
    # Background bar
    cv2.rectangle(img, (10, bar_y), (10 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    # Progress bar
    cv2.rectangle(img, (10, bar_y), (10 + int(bar_width * progress), bar_y + bar_height), (0, 200, 0), -1)
    
    return img


def view_demo(demo_file: str, demo_idx: int = 0, fps: int = 20, scale: int = 4, 
              start_step: int = 0, loop: bool = True):
    """View a demonstration with overlaid information.
    
    Args:
        demo_file: Path to HDF5 demo file
        demo_idx: Index of demo to view
        fps: Playback frames per second
        scale: Scale factor for display
        start_step: Step to start from
        loop: Whether to loop playback
    """
    print(f"Loading demo file: {demo_file}")
    
    with h5py.File(demo_file, 'r') as f:
        num_demos = f.attrs['num_demos']
        condition = f.attrs.get('condition', 'unknown')
        has_images = f.attrs.get('has_images', True)
        
        print(f"File info:")
        print(f"  Condition: {condition}")
        print(f"  Number of demos: {num_demos}")
        print(f"  Has images: {has_images}")
        
        if demo_idx >= num_demos:
            print(f"Error: demo_idx {demo_idx} >= num_demos {num_demos}")
            return
        
        if not has_images:
            print("Error: This demo file doesn't have images!")
            return
        
        # Load demo data
        demo_grp = f[f'demo_{demo_idx}']
        observations = demo_grp['observations'][:]
        actions = demo_grp['actions'][:]
        rewards = demo_grp['rewards'][:]
        
        # Load all available camera images
        camera_images = {}
        camera_names = []
        
        # Check for camera_names attribute (new format)
        if 'camera_names' in f.attrs:
            camera_names = f.attrs['camera_names'].split(',')
        
        # Load primary images
        if 'images' in demo_grp:
            camera_images['camera'] = demo_grp['images'][:]
            if 'camera' not in camera_names:
                camera_names.insert(0, 'camera')
        
        # Load additional camera images
        for key in demo_grp.keys():
            if key.startswith('images_'):
                cam_name = key.replace('images_', '')
                camera_images[cam_name] = demo_grp[key][:]
                if cam_name not in camera_names:
                    camera_names.append(cam_name)
        
        if not camera_images:
            print("Error: No images found in demo!")
            return
        
        # Use first camera as reference for shape
        images = camera_images[camera_names[0]]
        
        length = demo_grp.attrs['length']
        total_reward = demo_grp.attrs['total_reward']
        
        print(f"\nDemo {demo_idx} info:")
        print(f"  Length: {length} steps")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Cameras: {camera_names}")
        for cam_name in camera_names:
            print(f"    {cam_name}: {camera_images[cam_name].shape}")
        print(f"  Observation shape: {observations.shape}")
        print(f"  Action shape: {actions.shape}")
        
        print(f"\nControls:")
        print(f"  SPACE - Pause/Resume")
        print(f"  LEFT/RIGHT - Step backward/forward (when paused)")
        print(f"  R - Restart from beginning")
        print(f"  Q/ESC - Quit")
        print(f"  +/- - Increase/decrease playback speed")
        print(f"  S - Save current frame as PNG")
        
        # Create window
        window_name = f"Demo {demo_idx} - {condition}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        step = start_step
        paused = False
        current_fps = fps
        cumulative_reward = np.sum(rewards[:step+1]) if step > 0 else 0
        
        while True:
            # Render frames for all cameras and combine side by side
            frames = []
            for cam_name in camera_names:
                cam_images = camera_images[cam_name]
                frame = render_frame(
                    cam_images[step], 
                    observations[step], 
                    actions[step],
                    float(rewards[step]),
                    step,
                    length,
                    cumulative_reward,
                    scale
                )
                # Add camera label at top
                cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (255, 255, 0), 2, cv2.LINE_AA)
                frames.append(frame)
            
            # Combine frames side by side
            if len(frames) > 1:
                combined_frame = np.hstack(frames)
            else:
                combined_frame = frames[0]
            
            cv2.imshow(window_name, combined_frame)
            
            # Wait for key
            wait_time = 1 if paused else int(1000 / current_fps)
            key = cv2.waitKey(wait_time) & 0xFF
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space - pause/resume
                paused = not paused
                print(f"{'Paused' if paused else 'Playing'} at step {step}")
            elif key == ord('r'):  # R - restart
                step = 0
                cumulative_reward = 0
                print("Restarted")
            elif key == 81 or key == ord('a'):  # Left arrow or A - step back
                if paused and step > 0:
                    step -= 1
                    cumulative_reward = float(np.sum(rewards[:step+1]))
            elif key == 83 or key == ord('d'):  # Right arrow or D - step forward
                if paused and step < length - 1:
                    step += 1
                    cumulative_reward = float(np.sum(rewards[:step+1]))
            elif key == ord('+') or key == ord('='):  # Increase speed
                current_fps = min(60, current_fps + 5)
                print(f"FPS: {current_fps}")
            elif key == ord('-'):  # Decrease speed
                current_fps = max(1, current_fps - 5)
                print(f"FPS: {current_fps}")
            elif key == ord('s'):  # Save frame
                save_path = f"frame_{condition}_demo{demo_idx}_step{step}.png"
                cv2.imwrite(save_path, combined_frame)
                print(f"Saved frame to {save_path}")
            
            # Advance step if not paused
            if not paused:
                step += 1
                if step < length:
                    cumulative_reward += float(rewards[step])
                
                if step >= length:
                    if loop:
                        step = 0
                        cumulative_reward = 0
                        print("Looping...")
                    else:
                        print("Demo finished")
                        break
        
        cv2.destroyAllWindows()


def list_demos(demo_file: str):
    """List all demos in a file with their statistics."""
    print(f"Loading demo file: {demo_file}")
    
    with h5py.File(demo_file, 'r') as f:
        num_demos = f.attrs['num_demos']
        condition = f.attrs.get('condition', 'unknown')
        
        print(f"\n{'='*60}")
        print(f"Demo File: {demo_file}")
        print(f"Condition: {condition}")
        print(f"Number of demos: {num_demos}")
        print(f"{'='*60}\n")
        
        print(f"{'Idx':<5} {'Length':<10} {'Reward':<12} {'Has Images':<12}")
        print("-" * 45)
        
        for i in range(num_demos):
            grp = f[f'demo_{i}']
            length = grp.attrs['length']
            total_reward = grp.attrs['total_reward']
            has_imgs = 'images' in grp
            
            print(f"{i:<5} {length:<10} {total_reward:<12.2f} {str(has_imgs):<12}")


def main():
    parser = argparse.ArgumentParser(description="View saved demonstrations with overlaid info")
    parser.add_argument("--demo_file", type=str, required=True, help="Path to HDF5 demo file")
    parser.add_argument("--demo_idx", type=int, default=0, help="Index of demo to view")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for display")
    parser.add_argument("--start_step", type=int, default=0, help="Step to start from")
    parser.add_argument("--no_loop", action="store_true", help="Don't loop playback")
    parser.add_argument("--list", action="store_true", help="List demos in file instead of viewing")
    
    args = parser.parse_args()
    
    if not Path(args.demo_file).exists():
        print(f"Error: Demo file not found: {args.demo_file}")
        return
    
    if args.list:
        list_demos(args.demo_file)
    else:
        view_demo(
            args.demo_file,
            demo_idx=args.demo_idx,
            fps=args.fps,
            scale=args.scale,
            start_step=args.start_step,
            loop=not args.no_loop,
        )


if __name__ == "__main__":
    main()
