# TODO list for the upcoming weeks
## General goal:
*Given an fbx (containing animation skeleton), tsv's (one with 3D marker positions, one with 6DOF ball data) <- previous things are per throw, tsv with handball penalty throw trajectories and corresponding release point with features <- latter stay consistent throughout*
1. Extract PoR and corresponding features for this point
2. Feed into extraction model, which yields best matching throw from
3. Retarget trajectory to goal (only spatial, not speed) so that spatial release point of mocap matches the recorded in league data
4. Save as throw in blender or unity (VR environment)
5. Evaluate how smooth transition between data sources is
6. Do same, but with input being video only (secondary task)
## Things that need to be done, starting from upcoming
First, decide if Blender can be used for VR too, if yes -> use Blender with Blender scripting only?
1. Work on Penalty Processing Pipeline to enhance the quality of the parameters
    - Decide what to do with throws that have a deflection
    - Incorporate a step which creates a csv containing for each throw the important attrbutes
        - those are determined by how we build our extraction model -> Start with: Acceleration, Angle, Speed, ...
2. Build models (kNN, 1 or 2 others for comparison) (either pytorch or sklearn) for matching mocap throw to league throw
    - Prerequisites for that:
        - Decide which features to use
        - Write a script to extract the Point of Release (PoR) (possibly multiple) from a tsv file containing the body markers and a tsv file containing the ball positions (6DOF)
            - For this, write a simple visualization script, which takes in the 3D tsv file containing body markers and the 6DOF tsv file containing the ball's center, plot points from 3D as small sphere, 0.5cm diameter. Plot Ball points as spheres with 58cm circumference, be able to play at 300Hz, be able to go forth and back (also by frame), see currently selected frame, if not too hard:(be able to pass frame in visualization to go to), be able to pan view
                - Check, if positions are plausible
                - Use this script to double check the PoR detecor script
            - Write script which puts 3D (markers) and 6DOF (ball) in one csv file for PoR script as input
            - PoR detection script should be able to detect multiple PoRs in one tsv file, by detecting the first frame where the ball moves away from the hand (stabilize this by expecting noise in distance measure between ball and hand)
                - Decide what distance measure is appropriate
                    - Average distance of all Hand points (Hand_in, Hand_out, Pinkytip, Indextip, Thumbtip) to ball center? -> Would be simple and pretty stable. 
                        - Find a good deviation-to-previous-frames threshold, over which the ball is considered released. Trigger atleast if average distance between hand points and ball center is greater than 12cm (look ahead if this was only noise, if not then accept as PoR)
                        - Include check for ball staying close to hand for some time -> setting state to ball_in_hand
                        - When ball_in_hand, PoR can be detected
                        - Use this to toggle between ball_in_hand values whenever ball is again picked up, then thrown, then picked up, etc.
                - Should return the frame, time, x, y, z, rotation (from Rotation matrix from Rot[0-8]), and attributes for extraction model
                - Manually double check with visualization script afterwards
        - Look at possible feature engineering
    - Model: Input = PoR features, Output = throw (and corresponding trajectory and features) which matches best
3. Function to retarget throw output by model to match the spatial release point of mocap. 
4. Write scripts that animate character with fbx's skeleton, animate ball by 6DOF tsv until PoR, then by retargeted league throw trajectory
5. Put this into one pipeline to automate for new recorded throws
6. Extend step 2 so that video is also possible -> needs as input videos and calibration files
    - needs to detect ball via object detection, find PoR (or do manually, but still need PoR features, cannot do by hand!), find matching throw
    - pass video (with calibration done -> need script to automatically write calibration into corresponding place) into MAMMA model
    - use resulting file to animate character via MAMMA Blender Add-On, animate ball to be at certain offset of hand until PoR, then again control by league data
