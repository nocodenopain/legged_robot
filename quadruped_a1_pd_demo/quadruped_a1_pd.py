from weakref import ref
import mujoco
import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import time
import mujoco.viewer
import matplotlib.pyplot as plt

class A1PD:
    def __init__(self, model, data, ref_amplitude, ref_t_period, ref_dt):
        self.model = model
        self.data = data
        self.ref_amplitude = ref_amplitude
        self.ref_t_period = ref_t_period
        self.ref_dt = ref_dt


        # ========== Viewer Key_Callback ========== 
        self.PAUSE = False
        def key_callback(keycode):
            if chr(keycode) == ' ':  # pause simulation
                self.PAUSE = not self.PAUSE
            elif chr(keycode) == '.':  # reset
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:] = self.model.key_qpos[:]
                self.data.ctrl[:] = self.model.key_ctrl[:]
                mujoco.mj_step(self.model, self.data)
            elif chr(keycode) == ',':  # start control
                print("#"*20, "Control Start", "#"*20)
                self.CONTROL = True

        # ========== Viewer UI Setting ========== 
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.cam.azimuth = 135.2
        self.viewer.cam.elevation = -6.8
        self.viewer.cam.distance = 2.0
        self.model.vis.scale.contactwidth = 0.1
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.map.force = 0.01
        model.vis.scale.forcewidth = 0.03

        # Init position at keyframe home
        data.qpos[:] = model.key_qpos[:]
        data.ctrl[:] = model.key_ctrl[:]
        self.default_pos = data.ctrl[:]

        # Kinematic reference
        self.ref_iter_ = 0
        self.ref_steps = int(self.ref_t_period / self.ref_dt)
        self.kin_ref = KinematicRef(self.ref_amplitude, self.ref_t_period, self.ref_dt)
        self.poses = self.kin_ref.make_kinematic_ref()
    
        # Control
        self.control_iter_ = 0
        self.CONTROL = False

        # Print Simulation Info
        print("="*20, f"Simulation Info", "="*20)
        print(f"Sim Timestep: {self.model.opt.timestep}")
        print(f"Control Timestep: {ref_dt}")

        print(self.poses.shape)
        self.kin_ref.plot_kinematic_ref(self.poses)

    def control(self):
        if self.CONTROL:
            self.control_iter_ += 1
            print(f"Control Iter: {self.control_iter_}")
            if self.control_iter_ > self.ref_dt / self.model.opt.timestep:
                self.data.ctrl[:] = self.poses[self.ref_iter_] + self.default_pos
                self.ref_iter_ += 1
                self.control_iter_ = 0
                print(f"Contorl Progress: {self.ref_iter_ / self.ref_steps :.1%}")
                if self.ref_iter_ >= self.ref_steps:
                    self.CONTROL = False
                    self.ref_iter_ = 0
                    print("#"*20, "Control Finished", "#"*20)
         
    def run(self):
        # while self.data.time < 100.0 and self.viewer.is_running():
        while self.viewer.is_running():
            step_start = time.time()
            if not self.PAUSE:
                self.viewer.cam.lookat = self.data.body('trunk').subtree_com
                self.control()
                    
                mujoco.mj_step(self.model, self.data)

            self.viewer.sync()
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

class KinematicRef():
    def __init__(self, amplitude, t_period, dt=1/50):
        self.amplitude = amplitude
        self.t_period = t_period
        self.dt = dt

    def sin_wave(self, t):
        _sin_wave = self.amplitude * np.sin(((2*np.pi)/self.t_period)*t)
        return _sin_wave

    def make_kinematic_ref(self):
        _step_num = int(self.t_period / self.dt)
        _steps = np.arange(_step_num)
        t = _steps * self.dt
        
        wave = self.sin_wave(t)
        # Commands for one step of an active front leg
        leg_cmd_block = np.concatenate(
            [np.zeros((_step_num, 1)),
            wave.reshape(_step_num, 1),
            -2*wave.reshape(_step_num, 1)],
            axis=1
        )

        # In one step cycle, both pairs of active legs have inactive and active phases
        step_cycle = np.concatenate([leg_cmd_block, leg_cmd_block, leg_cmd_block, leg_cmd_block], axis=1)
        return step_cycle
    
    def plot_kinematic_ref(self, poses):
        # Plot curves of three joints on FR leg
        plt.figure(figsize=(10, 6))
        plt.plot(poses[:, 0], label='FR-abduction-0')
        plt.plot(poses[:, 1], label='FR-hip-1')
        plt.plot(poses[:, 2], label='FR-knee-2')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Kinematic Reference Positions')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    
    model = mujoco.MjModel.from_xml_path("D:/code/genesis_test/quadruped_a1_pd_demo/a1_mjcf/scene.xml")
    data = mujoco.MjData(model)
    ref_amplitude = 0.01
    ref_t_period = 1
    ref_dt = 1/50

    A1Sim = A1PD(model, data, ref_amplitude, ref_t_period, ref_dt)
    A1Sim.run()