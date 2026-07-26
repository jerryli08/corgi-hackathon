# Paste this whole block when you're back.
# Before recording: aim C920 so the pill bottle is clearly in frame.

# Keep the old 5-episode test; write a fresh set here:
#   datasets/walker_pill_grasp_v2

lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=walker_follower `
  --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
  --teleop.type=so101_leader `
  --teleop.port=COM10 `
  --teleop.id=walker_leader `
  --dataset.repo_id=local/walker_pill_grasp_v2 `
  --dataset.root=datasets/walker_pill_grasp_v2 `
  --dataset.num_episodes=5 `
  --dataset.episode_time_s=60 `
  --dataset.reset_time_s=30 `
  --dataset.single_task="grasp the pill bottle on the shelf" `
  --dataset.push_to_hub=false `
  --display_data=false `
  --dataset.streaming_encoding=true `
  --dataset.encoder_threads=2
