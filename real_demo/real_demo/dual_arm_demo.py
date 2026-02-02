import os
import time

import mujoco as mj
from mujoco import viewer

import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.insert(0, PROJECT_ROOT)
from real_demo.sampling_based_planner.cem_planner import CemPlanner
from real_demo.sampling_based_planner.quat_math import *

PACKAGE_DIR = '/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/11_MPC/bimanual_manipulation/real_demo'
np.set_printoptions(precision=4, suppress=True)

target_positions = np.array([
    [-0.25, -0.2, 0.3],
    [-0.25, 0.0, 0.3],
    [-0.25, -0.1, 0.3],
])

target_rotations = np.array([
    quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-90, [0, 0, 1])),
    # quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-135, [0, 0, 1])),
    quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-45, [0, 0, 1])),
    np.array([1, 0, 0, 0])
])


class Planner:
    def __init__(self):
        # Demo params
        self.use_hardware = False
        self.record_data_ = False
        self.idx = 0
        self.idx = str(self.idx).zfill(2)

        # Planner params
        self.num_dof = 12
        self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
        num_batch = 500
        num_steps = 15
        maxiter_cem = 1
        maxiter_projection = 5
        # w_pos= 3.0
        # w_rot= 0.5
        # w_col= 500.0
        num_elite = 0.05
        self.timestep = 0.1
        position_threshold = 0.06
        rotation_threshold = 0.1

        self.num_targets = 21

        if self.record_data_:
            self.pathes = {
                "setup": os.path.join(PACKAGE_DIR, 'data', 'planner', 'setup', f'setup_{self.idx}.npz'),
                "trajectory": os.path.join(PACKAGE_DIR, 'data', 'planner', 'trajectory', f'traj_{self.idx}.npz'),
                "benchmark": os.path.join(PACKAGE_DIR, 'data', 'planner', 'benchmark',
                                          f'bench_{num_batch}_{num_steps}_19{self.idx}.npz'),
            }
            self.data_buffers = {
                'batch_size': [num_batch],
                'horizon': [num_steps],

                'target_0': [0] * self.num_targets,
                'total_time_s': [0] * self.num_targets,
                'success': [0] * self.num_targets,
                'reason': [0] * self.num_targets,

                'step_time_ms': [[] for _ in range(self.num_targets)],
                'theta': [[] for _ in range(self.num_targets)],
                'thetadot': [[] for _ in range(self.num_targets)],

                'cost_r': [[] for _ in range(self.num_targets)],
                'cost_eef_to_obj': [[] for _ in range(self.num_targets)],
                'cost_obj_to_targ': [[] for _ in range(self.num_targets)],
                'cost_dist': [[] for _ in range(self.num_targets)],
                'cost_zy': [[] for _ in range(self.num_targets)],
            }

        self.task = 'pick'

        cost_weights = {
            'collision': 500,
            'theta': 0.3,
            'z-axis': 5.0,
            'velocity': 0.1,

            'position': 3.0,
            'orientation_pick': 0.5,

            'distance': 20.0,
            'position_tray': 10.0,
            'orientation_tray': 2,
            'orientation_move': 10,

            'pick': 0,
            'move': 0
        }

        self.grab_pos_thresh = 0.02
        self.grab_rot_thresh = 0.05
        self.thetadot = np.zeros(self.num_dof)

        self.grippers = {
            '0': {
                'srv': None,
                'state': 'open'
            },
            '1': {
                'srv': None,
                'state': 'open'
            }
        }

        # Initialize MuJoCo model and data
        model_path = f'{PACKAGE_DIR}/ur5e_hande_mjx/scene.xml'
        self.model: mj.MjModel = mj.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.timestep

        self.data = mj.MjData(self.model)

        joint_names_pos = list()
        joint_names_vel = list()
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            n_pos = 7 if joint_type == mj.mjtJoint.mjJNT_FREE else 4 if joint_type == mj.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mj.mjtJoint.mjJNT_FREE else 3 if joint_type == mj.mjtJoint.mjJNT_BALL else 1

            for _ in range(n_pos):
                joint_names_pos.append(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i))

        robot_joints = np.array(
            ['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1',
             'wrist_3_joint_1',
             'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2',
             'wrist_3_joint_2'])

        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position

        self.gripper_0_act_idx = self.model.actuator('fingers_actuator_0').id
        self.gripper_1_act_idx = self.model.actuator('fingers_actuator_1').id

        target_0_rot = quaternion_multiply(
            quaternion_multiply(self.model.body(name="target_0").quat, rotation_quaternion(-180, [0, 1, 0])),
            rotation_quaternion(-90, [0, 0, 1]))
        target_1_rot = quaternion_multiply(
            quaternion_multiply(self.model.body(name="target_1").quat, rotation_quaternion(180, [0, 1, 0])),
            rotation_quaternion(90, [0, 0, 1]))
        # target_2_rot = quaternion_multiply(self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]], rotation_quaternion(-45, [0, 0, 1]))

        self.model.body(name='target_0').quat = target_0_rot
        self.model.body(name='target_1').quat = target_1_rot
        self.model.body(name='target_00').quat = target_0_rot
        self.model.body(name='target_11').quat = target_1_rot
        # self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = target_2_rot

        # Set the table positions alligmed with the motion capture coordinate system

        # setup = np.load(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.npz'), allow_pickle=True)

        # marker_pos = setup['setup'][0][1]
        # marker_diff = marker_pos-self.model.body(name='table0_marker').pos

        # self.model.body(name='table_0').pos = setup['setup'][0][0]
        # self.model.body(name='table0_marker').pos = setup['setup'][0][1]
        # self.model.body(name='table_1').pos = setup['setup'][0][2]
        # self.model.body(name='table1_marker').pos = setup['setup'][0][3]

        # self.model.body(name='tray').pos += marker_diff
        # self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] += marker_diff
        # self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] += marker_diff

        table_0_pos = self.model.body(name='table_0').pos
        table_1_pos = self.model.body(name='table_1').pos

        # if self.use_hardware:
        #     setup = np.load(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.npz'), allow_pickle=True)

        #     marker_pos = setup['setup'][0][1]
        #     marker_diff = marker_pos-self.model.body(name='table0_marker').pos

        #     self.model.body(name='table_0').pos = setup['setup'][0][0]
        #     self.model.body(name='table0_marker').pos = setup['setup'][0][1]
        #     self.model.body(name='table_1').pos = setup['setup'][0][2]
        #     self.model.body(name='table1_marker').pos = setup['setup'][0][3]

        #     self.model.body(name='tray').pos += marker_diff
        #     self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] += marker_diff
        #     self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] += marker_diff
        # else:
        #     table_0_pos = self.model.body(name='table_0').pos
        #     table_1_pos = self.model.body(name='table_1').pos

        mj.mj_forward(self.model, self.data)

        self.tray_init_pos = np.concatenate(
            [self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]],
             self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]])

        self.success = 0
        self.reason = 'na'
        self.traj_time_start = time.time()
        self.target_idx = 0

        # Initialize CEM/MPC planner
        self.planner = CemPlanner(
            model=self.model,
            data=self.data,
            num_dof=self.num_dof,
            num_batch=num_batch,
            num_steps=num_steps,
            maxiter_cem=maxiter_cem,
            maxiter_projection=maxiter_projection,
            num_elite=num_elite,
            timestep=self.timestep,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            table_0_pos=table_0_pos,
            table_1_pos=table_1_pos,
            cost_weights=cost_weights
        )

        # Setup viewer
        self.viewer = mj.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = self.model.body(name='table_0').pos
        self.viewer.cam.distance = 5.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -30.0

        # self.viewer.cam.frame_size = 0.03  # default is usually 0.1 

    def render_trace(self, viewer_, *eef_trace_positions):

        """Render the end-effector trajectory trace in the viewer."""
        # Clear any existing overlay geoms
        viewer_.user_scn.ngeom = 0
        for trace in eef_trace_positions:
            # Add spheres for each position in the trace
            for pos in trace:
                # Create a new geom in the user scene
                geom_id = viewer_.user_scn.ngeom
                viewer_.user_scn.ngeom += 1

                # Initialize the geom properties
                mj.mjv_initGeom(
                    viewer_.user_scn.geoms[geom_id],
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0.01, 0.01],  # radius 1 cm sphere
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 1, 0.5]
                )

    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()

        if self.task == 'move':
            eef_pos_0 = self.data.site_xpos[self.planner.tcp_id_0]
            eef_pos_1 = self.data.site_xpos[self.planner.tcp_id_1]

            tray_pos = (eef_pos_0 + eef_pos_1) / 2 - np.array([0, 0, 0.1])
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos

            tray_rot_init = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
            tray_0_pos = self.data.xpos[self.model.body(name='target_0').id]
            tray_1_pos = self.data.xpos[self.model.body(name='target_1').id]
            tray_rot = turn_quat(tray_0_pos, tray_1_pos, eef_pos_0, eef_pos_1, tray_rot_init)
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_rot

            self.planner.update_targets(target_idx=0, target_pos=self.data.xpos[self.model.body(name="target_00").id],
                                        target_rot=self.data.xquat[self.model.body(name="target_00").id])
            self.planner.update_targets(target_idx=1, target_pos=self.data.xpos[self.model.body(name="target_11").id],
                                        target_rot=self.data.xquat[self.model.body(name="target_11").id])

        # Get current state
        if self.use_hardware:
            current_pos_0 = np.array(self.rtde_r_0.getActualQ())
            current_pos_1 = np.array(self.rtde_r_1.getActualQ())

            current_pos = np.concatenate((current_pos_0, current_pos_1), axis=None)
            current_vel = self.thetadot
        else:
            current_pos = self.data.qpos[self.joint_mask_pos]
            current_vel = self.thetadot

        # Compute control
        self.thetadot, cost, cost_list, thetadot_horizon, theta_horizon, eef_0_planned, eef_1_planned = (
            self.planner.compute_control(current_pos, current_vel, self.task))
        cost_c, cost_dist, cost_g, cost_r = cost_list

        self.data.qvel[:] = np.zeros(len(self.joint_mask_vel))
        self.data.qvel[self.joint_mask_vel] = self.thetadot
        mj.mj_step(self.model, self.data)

        self.render_trace(self.viewer, eef_0_planned[:, :3], eef_1_planned[:, :3])

        current_cost_g_0 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_0] - self.planner.target_0[:3])
        current_cost_r_0 = quaternion_distance(self.data.xquat[self.planner.hande_id_0], self.planner.target_0[3:])

        current_cost_g_1 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_1] - self.planner.target_1[:3])
        current_cost_r_1 = quaternion_distance(self.data.xquat[self.planner.hande_id_1], self.planner.target_1[3:])

        tray_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
        tray_rot = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
        current_cost_g_tray = np.linalg.norm(tray_pos - self.planner.target_2[:3])
        current_cost_r_tray = quaternion_distance(tray_rot, self.planner.target_2[3:])

        distance = np.linalg.norm(
            self.data.site_xpos[self.planner.tcp_id_0] - self.data.site_xpos[self.planner.tcp_id_1])
        cost_dist_s = np.abs(distance - 0.30)

        cost_z_s = np.abs(self.data.site_xpos[self.planner.tcp_id_0][2] - self.data.site_xpos[self.planner.tcp_id_1][2])

        cost_r_s = np.mean([quaternion_distance(self.data.xquat[self.planner.hande_id_0],
                                                self.data.xquat[self.model.body(name="target_0").id]),
                            quaternion_distance(self.data.xquat[self.planner.hande_id_1],
                                                self.data.xquat[self.model.body(name="target_1").id])])
        cost_g_s = np.mean([np.linalg.norm(
            self.data.site_xpos[self.planner.tcp_id_0] - self.data.xpos[self.model.body(name="target_0").id]),
            np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_1] - self.data.xpos[
                self.model.body(name="target_1").id])])

        target_reached = False
        if self.task == 'pick':
            target_reached = (
                    current_cost_g_0 < self.grab_pos_thresh \
                    and current_cost_r_0 < self.grab_rot_thresh \
                    and current_cost_g_1 < self.grab_pos_thresh \
                    and current_cost_r_1 < self.grab_rot_thresh
            )
        elif self.task == 'move':
            target_reached = (
                    current_cost_g_tray < self.grab_pos_thresh \
                    and current_cost_r_tray < self.grab_rot_thresh \
                )

        if target_reached and self.task == 'pick':
            self.task = 'move'
            self.gripper_control(gripper_act_idx=self.gripper_0_act_idx, action=0)
            self.gripper_control(gripper_act_idx=self.gripper_1_act_idx, action=0)
        elif target_reached and self.task == 'move':
            print("================== TARGRT REACHED UPDATING TARGET ==================", flush=True)
            self.success = 1
            self.reason = 'na'
            self.reset_simulation()

        if self.task == 'move' and cost_dist_s > 0.1:
            print("================== FAILED: DISTANCE ==================", flush=True)
            self.success = 0
            self.reason = 'dist'
            self.reset_simulation()
        if self.task == 'move' and cost_z_s > 0.05:
            print("================== FAILED: Z ==================", flush=True)
            self.success = 0
            self.reason = 'z'
            self.reset_simulation()

        if self.task == 'move' and cost_r_s > 0.2:
            print("================== FAILED: Z ==================", flush=True)
            self.success = 0
            self.reason = 'rotation'
            self.reset_simulation()
        if cost_c > 300:
            print("================== FAILED: COLLISION ==================", flush=True)
            self.success = 0
            self.reason = 'collision'
            self.reset_simulation()

        if time.time() - self.traj_time_start > 60:
            print("======================= TARGET FAILED: TIMEOUT =======================", flush=True)
            self.success = 0
            self.reason = 'timeout'
            self.reset_simulation()

        if self.record_data_ and self.target_idx < self.num_targets:
            theta = self.data.qpos[self.joint_mask_pos]
            step_time_ms = (time.time() - start_time) * 1000

            self.data_buffers['step_time_ms'][self.target_idx].append(step_time_ms)
            self.data_buffers['theta'][self.target_idx].append(theta.copy())
            self.data_buffers['thetadot'][self.target_idx].append(self.thetadot.copy())

            self.data_buffers['cost_r'][self.target_idx].append(cost_r_s.copy())
            self.data_buffers['cost_eef_to_obj'][self.target_idx].append(cost_g_s.copy())
            self.data_buffers['cost_obj_to_targ'][self.target_idx].append(current_cost_g_tray.copy())
            self.data_buffers['cost_dist'][self.target_idx].append(cost_dist_s.copy())
            self.data_buffers['cost_zy'][self.target_idx].append(cost_z_s.copy())

        # Update viewer
        self.viewer.sync()

        # Print debug info
        print(f'\n| Target idx: {self.target_idx} '
              f'\n| Task: {self.task} '
              f'\n| Total Time: {"%.0f" % (time.time() - self.traj_time_start)}ms '
              f'\n| Step Time: {"%.0f" % ((time.time() - start_time) * 1000)}ms '
              f'\n| Cost dist: {"%.2f, %.2f" % (float(cost_dist), float(cost_dist_s))} '
              f'\n| Cost z: {"%.2f" % (float(cost_z_s))} '
              f'\n| Cost r: {"%.2f" % (float(cost_r_s))} '
              f'\n| Cost g mjx: {"%.2f" % (float(cost_g))} '
              f'\n| Cost r mjx: {"%.2f" % (float(cost_r))} '
              f'\n| Cost c: {"%.2f" % (float(cost_c))} '
              f'\n| Cost gr0: {"%.2f, %.2f" % (float(current_cost_g_0), float(current_cost_r_0))} '
              f'\n| Cost gr1: {"%.2f, %.2f" % (float(current_cost_g_1), float(current_cost_r_1))} '
              f'\n| Cost tr: {"%.2f, %.2f" % (float(current_cost_g_tray), float(current_cost_r_tray))} '
              f'\n| Cost: {np.round(cost, 2)} ', flush=True)

        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def reset_simulation(self):
        if self.record_data_ and self.target_idx < self.num_targets:
            self.data_buffers['success'][self.target_idx] = self.success
            self.data_buffers['reason'][self.target_idx] = self.reason
            self.data_buffers['total_time_s'][self.target_idx] = (time.time() - self.traj_time_start)
            self.data_buffers['target_0'][self.target_idx] = self.planner.target_2.copy()

        self.success = 0
        self.reason = 'na'
        self.traj_time_start = time.time()
        self.target_idx += 1

        self.task = 'pick'
        self.planner.xi_cov = np.kron(np.eye(self.planner.cem.num_dof), 10 * np.identity(self.planner.cem.nvar_single))
        self.planner.xi_mean = np.zeros(self.planner.cem.nvar)
        self.data.qpos[self.joint_mask_pos] = self.init_joint_position
        self.data.qvel[self.joint_mask_vel] = np.zeros(self.init_joint_position.shape)
        self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = self.tray_init_pos[:3]
        self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = self.tray_init_pos[3:]

        target_pos, target_rot = self.generate_targets()
        self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] = target_pos
        self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] = target_rot

        mj.mj_step(self.model, self.data)

        self.planner.target_0 = np.concatenate(
            [self.data.xpos[self.model.body(name="target_0").id], self.data.xquat[self.model.body(name="target_0").id]])
        self.planner.target_1 = np.concatenate(
            [self.data.xpos[self.model.body(name="target_1").id], self.data.xquat[self.model.body(name="target_1").id]])

        self.planner.target_2[:3] = target_pos
        self.planner.target_2[3:] = target_rot

    def generate_targets(self):
        area_center_1 = np.array([-0.25, -0.1, 0.3])
        area_size_1 = np.array([0.15, 0.15, 0.1])

        target_pos = area_center_1 + np.random.uniform(-area_size_1, area_size_1, size=3)
        target_rot = target_rotations[np.random.randint(0, 3)]
        return target_pos, target_rot

    def gripper_control(self, gripper_act_idx, action=255):
        self.grippers[str(gripper_act_idx)]['state'] = action
        self.data.ctrl[gripper_act_idx] = action
        print(f"Gripper {gripper_act_idx} has complited {action} action.")

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.gripper_control(gripper_act_idx=self.gripper_0_act_idx, action=255)
        self.gripper_control(gripper_act_idx=self.gripper_1_act_idx, action=255)
        print("Moved to initial pose.", flush=True)

    def object0_callback(self, msg):
        """Callback for target object pose updates"""

        if self.task == 'pick':
            pose = msg.pose
            tray_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z - 0.08])
            self.model.body(name='tray').pos = tray_pos
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos
            mj.mj_forward(self.model, self.data)
            self.planner.update_targets(target_idx=0, target_pos=self.data.xpos[self.model.body(name="target_0").id],
                                        target_rot=self.model.body(name='target_0').quat)
            self.planner.update_targets(target_idx=1, target_pos=self.data.xpos[self.model.body(name="target_1").id],
                                        target_rot=self.model.body(name='target_1').quat)

    def obstacle0_callback(self, msg):
        """Callback for obstacle pose updates"""
        pose = msg.pose
        obstacle_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z])
        obstacle_rot = np.array([0.0, 1.0, 0, 0])
        self.planner.update_obstacle(obstacle_pos, obstacle_rot)

    def record_data(self):
        """Save data to npy file"""
        # self.data_buffers['setup'].append([self.model.body(name='table_0').pos, self.model.body(name='table0_marker').pos, 
        #                                    self.model.body(name='table_1').pos, self.model.body(name='table1_marker').pos])
        # np.savez(
        #     self.pathes['setup'],
        #     setup=self.data_buffers['setup'],
        # )
        # np.savez(
        #     self.pathes['trajectory'],
        #     theta=np.array(self.data_buffers['theta']),
        #     thetadot=np.array(self.data_buffers['thetadot']),
        #     theta_planned=np.array(self.data_buffers['theta_planned']),
        #     thetadot_planned=np.array(self.data_buffers['thetadot_planned']),
        #     target_0=np.array(self.data_buffers['target_0']),
        #     target_1=np.array(self.data_buffers['target_1']),
        #     theta_planned_batched=np.array(self.data_buffers['theta_planned_batched']),
        #     thetadot_planned_batched=np.array(self.data_buffers['thetadot_planned_batched']),
        #     cost_cgr_batched=np.array(self.data_buffers['cost_cgr_batched']),
        #     timestamp=np.array(self.data_buffers['timestamp']),
        # )
        np.savez(
            self.pathes['benchmark'],
            batch_size=np.array(self.data_buffers['batch_size']),
            horizon=np.array(self.data_buffers['horizon']),
            total_time=np.array(self.data_buffers['total_time_s']),
            step_time=np.array(self.data_buffers['step_time_ms'], dtype=object),
            success=np.array(self.data_buffers['success']),
            reason=np.array(self.data_buffers['reason']),
            target_0=np.array(self.data_buffers['target_0'], dtype=object),
            theta=np.array(self.data_buffers['theta'], dtype=object),
            thetadot=np.array(self.data_buffers['thetadot'], dtype=object),
            cost_r=np.array(self.data_buffers['cost_r'], dtype=object),
            cost_eef_to_obj=np.array(self.data_buffers['cost_eef_to_obj'], dtype=object),
            cost_obj_to_targ=np.array(self.data_buffers['cost_obj_to_targ'], dtype=object),
            cost_dist=np.array(self.data_buffers['cost_dist'], dtype=object),
            cost_zy=np.array(self.data_buffers['cost_zy'], dtype=object),
        )
        self.data_saved = True
        print("Saving data...")


def main(args=None):
    planner = Planner()
    print("Initialized node.", flush=True)

    try:
        while planner.viewer.is_running():
            planner.control_loop()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        planner.viewer.close()
        if planner.record_data_:
            planner.record_data()


if __name__ == '__main__':
    main()
