import sys
import time
import hppfcl as fcl
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

RED_COLOR = np.array([1.0, 0.0, 0.0, 1.0])
BLUE_COLOR = np.array([0.0, 0.0, 1.0, 1.0])
WHITE_COLOR = np.array([1.0, 1.0, 1.0, 1.0])
GREEN_COLOR = np.array([0.0, 1.0, 0.0, 1.0])
ORANGE_COLOR = np.array([1.0, 0.5, 0.0, 1.0])

class ParallelLegArm:
    """
    This class simulates a closed-loop five-bar linkage similar to the Minitaur
    robot leg. The mechanism has two parallel motors at the top (hip), each 
    driving a support arm. The support arms connect to main links that hang 
    downward and meet at a common foot point, forming a parallel linkage.
    
    The leg has two kinematic chains originating from parallel motors at the hip:
    
    Chain A (primary): Motor_A -> Support_A -> Main_Link -> Foot
    Chain B (secondary): Motor_B -> Support_B -> Secondary_Link -> (attaches to Main_Link)
    
    Both motors are at the same location (base), but rotate independently.
    The chains meet via a loop closure constraint.
    """

    def __init__(
        self,
        length_support: float = 100 / 1000,
        length_main: float = 199 / 1000,
        length_foot: float = 86 / 1000,
        length_secondary: float = 199 / 1000,
        mass_support: float = 0.33,
        mass_main: float = 0.043,
        mass_foot: float = 0.048,
        mass_secondary: float = 0.045,
        height: float = 0.05,
        width: float = 0.01,
        radius: float = 0.01
    ):
        """
        Initialize the parallel leg arm mechanism.
        
        Args:
            length_support: Length of support arms (m)
            length_main: Length of main link upper portion (from knee to constraint point) (m)
            length_foot: Length of foot portion (from constraint point to tip) (m)
            length_secondary: Length of secondary link (m)
            mass_support: Mass of support arms (kg)
            mass_main: Mass of main link upper portion (kg)
            mass_foot: Mass of foot portion (kg)
            mass_secondary: Mass of secondary link (kg)
            height: Cross-section height of links (m)
            width: Cross-section width of links (m)
            radius: Radius of capsule collision shapes (m)
        """
        # Store parameters
        self.length_support = length_support
        self.length_main = length_main
        self.length_foot = length_foot
        self.length_secondary = length_secondary
        self.mass_support = mass_support
        self.mass_main = mass_main
        self.mass_foot = mass_foot
        self.mass_secondary = mass_secondary
        self.height = height
        self.width = width
        self.radius = radius
        
        # Build the model
        self._build_model()
        
        # Solve initial configuration
        self.q = self._solve_initial_ik()
        self.v = np.zeros(self.model.nv)
        self.a = np.zeros(self.model.nv)
        
        # Smoothing factor for velocity/acceleration (0 = no smoothing, 1 = infinite smoothing)
        self.smoothing = 0.9
        
        # Visualization (initialized later if needed)
        self.viz = None
        
    def _create_link_properties(self, mass: float, length: float):
        """Create inertia and placements for a link."""
        inertia = pin.Inertia.FromBox(mass, length, self.width, self.height)
        placement_center = pin.SE3.Identity()
        placement_center.translation = pin.XAxis * length / 2.0
        placement_shape = placement_center.copy()
        placement_shape.rotation = pin.Quaternion.FromTwoVectors(pin.ZAxis, pin.XAxis).matrix()
        return inertia, placement_center, placement_shape

    def _build_model(self):
        """Build the Pinocchio model and geometry."""
        # Create collision shapes
        shape_support = fcl.Capsule(self.radius, self.length_support)
        shape_main = fcl.Capsule(self.radius, self.length_main)
        shape_foot = fcl.Capsule(self.radius, self.length_foot)
        shape_secondary = fcl.Capsule(self.radius, self.length_secondary)
        
        # Create inertia properties
        inertia_support, _, _ = self._create_link_properties(self.mass_support, self.length_support)
        inertia_main, _, _ = self._create_link_properties(self.mass_main, self.length_main)
        inertia_foot, _, _ = self._create_link_properties(self.mass_foot, self.length_foot)
        inertia_secondary, _, _ = self._create_link_properties(self.mass_secondary, self.length_secondary)
        
        # Initialize models
        self.model = pin.Model()
        self.collision_model = pin.GeometryModel()
        
        base_joint_id = 0
        
        # Motor A: First revolute joint at base (Y-axis rotation)
        motor_A_placement = pin.SE3.Identity()
        self.motor_A_id = self.model.addJoint(
            base_joint_id, pin.JointModelRY(), motor_A_placement, "motor_A"
        )
        
        # Support Arm A: Short link from Motor A, pointing downward initially
        support_A_rotation = pin.Quaternion.FromTwoVectors(pin.XAxis, -pin.ZAxis).matrix()
        self.model.appendBodyToJoint(
            self.motor_A_id, inertia_support, 
            pin.SE3(support_A_rotation, -pin.ZAxis * self.length_support / 2.0)
        )
        
        geom_support_A = pin.GeometryObject(
            "support_A", self.motor_A_id,
            pin.SE3(pin.Quaternion.FromTwoVectors(pin.ZAxis, -pin.ZAxis).matrix(), 
                   -pin.ZAxis * self.length_support / 2.0),
            shape_support
        )
        geom_support_A.meshColor = RED_COLOR
        self.collision_model.addGeometryObject(geom_support_A)
        
        # Knee joint A: Connects Support A to Main Link
        knee_A_placement = pin.SE3.Identity()
        knee_A_placement.translation = -pin.ZAxis * self.length_support
        self.knee_A_id = self.model.addJoint(
            self.motor_A_id, pin.JointModelRY(), knee_A_placement, "knee_A"
        )
        
        # Main Link: From knee A to constraint attachment point
        main_rotation = pin.Quaternion.FromTwoVectors(pin.XAxis, -pin.ZAxis).matrix()
        self.model.appendBodyToJoint(
            self.knee_A_id, inertia_main, 
            pin.SE3(main_rotation, -pin.ZAxis * self.length_main / 2.0)
        )
        
        # Foot Portion: From constraint attachment point to tip
        foot_placement = pin.SE3.Identity()
        foot_placement.translation = -pin.ZAxis * (self.length_main + self.length_foot / 2.0)
        self.model.appendBodyToJoint(
            self.knee_A_id, inertia_foot, 
            pin.SE3(main_rotation, foot_placement.translation)
        )
        
        # Geometry for main link
        geom_main = pin.GeometryObject(
            "main_link", self.knee_A_id,
            pin.SE3(pin.Quaternion.FromTwoVectors(pin.ZAxis, -pin.ZAxis).matrix(), 
                   -pin.ZAxis * self.length_main / 2.0),
            shape_main
        )
        geom_main.meshColor = WHITE_COLOR
        self.collision_model.addGeometryObject(geom_main)
        
        # Geometry for foot portion
        geom_foot = pin.GeometryObject(
            "foot", self.knee_A_id,
            pin.SE3(pin.Quaternion.FromTwoVectors(pin.ZAxis, -pin.ZAxis).matrix(), 
                   -pin.ZAxis * (self.length_main + self.length_foot / 2.0)),
            shape_foot
        )
        geom_foot.meshColor = ORANGE_COLOR
        self.collision_model.addGeometryObject(geom_foot)
        
        # Motor B: Second revolute joint at base (parallel to Motor A)
        motor_B_placement = pin.SE3.Identity()
        self.motor_B_id = self.model.addJoint(
            base_joint_id, pin.JointModelRY(), motor_B_placement, "motor_B"
        )
        
        # Support Arm B: Short link from Motor B
        self.model.appendBodyToJoint(
            self.motor_B_id, inertia_support, 
            pin.SE3(support_A_rotation, -pin.ZAxis * self.length_support / 2.0)
        )
        
        geom_support_B = pin.GeometryObject(
            "support_B", self.motor_B_id,
            pin.SE3(pin.Quaternion.FromTwoVectors(pin.ZAxis, -pin.ZAxis).matrix(), 
                   -pin.ZAxis * self.length_support / 2.0),
            shape_support
        )
        geom_support_B.meshColor = BLUE_COLOR
        self.collision_model.addGeometryObject(geom_support_B)
        
        # Knee joint B: Connects Support B to Secondary Link
        knee_B_placement = pin.SE3.Identity()
        knee_B_placement.translation = -pin.ZAxis * self.length_support
        self.knee_B_id = self.model.addJoint(
            self.motor_B_id, pin.JointModelRY(), knee_B_placement, "knee_B"
        )
        
        # Secondary Link: Connects from knee B to a point on the main link
        secondary_rotation = pin.Quaternion.FromTwoVectors(pin.XAxis, -pin.ZAxis).matrix()
        self.model.appendBodyToJoint(
            self.knee_B_id, inertia_secondary, 
            pin.SE3(secondary_rotation, -pin.ZAxis * self.length_secondary / 2.0)
        )
        
        geom_secondary = pin.GeometryObject(
            "secondary_link", self.knee_B_id,
            pin.SE3(pin.Quaternion.FromTwoVectors(pin.ZAxis, -pin.ZAxis).matrix(), 
                   -pin.ZAxis * self.length_secondary / 2.0),
            shape_secondary
        )
        geom_secondary.meshColor = GREEN_COLOR
        self.collision_model.addGeometryObject(geom_secondary)
        
        # Use collision model as visual model
        self.visual_model = self.collision_model
        
        # Create data structures
        self.data = self.model.createData()
        
        # Setup loop closure constraint
        self._setup_constraint()
        
    def _setup_constraint(self):
        """Setup the loop closure constraint between secondary link and main link."""
        # Constraint attachment point on knee_B joint (end of secondary link)
        constraint1_joint1_placement = pin.SE3.Identity()
        constraint1_joint1_placement.translation = -pin.ZAxis * self.length_secondary
        
        # Constraint attachment point on knee_A joint (at end of main link, where foot begins)
        constraint1_joint2_placement = pin.SE3.Identity()
        constraint1_joint2_placement.translation = -pin.ZAxis * self.length_main
        
        # Create the rigid constraint model (3D point-to-point contact)
        self.constraint_model = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            self.model,
            self.knee_B_id,
            constraint1_joint1_placement,
            self.knee_A_id,
            constraint1_joint2_placement,
        )
        self.constraint_data = self.constraint_model.createData()
        self.constraint_dim = self.constraint_model.size()
        
        # Baumgarte stabilization gains
        self.constraint_model.corrector.Kd[:] = 100 / (2.0 * np.sqrt(10))
        
    def _solve_initial_ik(self, motor_A_angle: float = 0.5, motor_B_angle: float = -0.5) -> np.ndarray:
        """
        Solve inverse kinematics to find a valid configuration.
        
        Args:
            motor_A_angle: Initial angle for motor A (rad)
            motor_B_angle: Initial angle for motor B (rad)
            
        Returns:
            Configuration vector q that satisfies constraints
        """
        # Initialize configuration
        q0 = pin.neutral(self.model)
        q0[0] = motor_A_angle   # Motor A angle
        q0[1] = -0.6            # Knee A angle (bent outward)
        q0[2] = motor_B_angle   # Motor B angle
        q0[3] = 0.6             # Knee B angle
        
        pin.forwardKinematics(self.model, self.data, q0)
        
        # Solver parameters
        rho = 1e-10
        mu = 1e-4
        eps = 1e-10
        N = 200
        
        q = q0.copy()
        y = np.ones(self.constraint_dim)
        
        self.data.M = np.eye(self.model.nv) * rho
        kkt_constraint = pin.ContactCholeskyDecomposition(self.model, [self.constraint_model])
        
        for k in range(N):
            pin.computeJointJacobians(self.model, self.data, q)
            kkt_constraint.compute(self.model, self.data, [self.constraint_model], [self.constraint_data], mu)
            
            constraint_value = self.constraint_data.c1Mc2.translation
            
            J = pin.getFrameJacobian(
                self.model,
                self.data,
                self.constraint_model.joint1_id,
                self.constraint_model.joint1_placement,
                self.constraint_model.reference_frame,
            )[:3, :]
            
            primal_feas = np.linalg.norm(constraint_value, np.inf)
            dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
            if primal_feas < eps and dual_feas < eps:
                break
            
            rhs = np.concatenate([-constraint_value - y * mu, np.zeros(self.model.nv)])
            dz = kkt_constraint.solve(rhs)
            dy = dz[:self.constraint_dim]
            dq = dz[self.constraint_dim:]
            
            alpha = 1.0
            q = pin.integrate(self.model, q, -alpha * dq)
            y -= alpha * (-dy + y)
        
        return q
    
    def update(self, motor_A_position: float, motor_B_position: float, dt: float = None) -> bool:
        """
        Update the arm configuration based on motor positions using inverse kinematics.
        
        This solves the loop closure constraint to find valid knee angles given
        the desired motor positions. If dt is provided, also computes joint velocities.
        
        Args:
            motor_A_position: Desired angle for motor A (rad)
            motor_B_position: Desired angle for motor B (rad)
            dt: Time step since last update (s). If provided, velocities are computed.
            
        Returns:
            True if IK converged successfully, False otherwise
        """
        # Store previous configuration for velocity computation
        q_prev = self.q.copy()
        
        # Set motor positions and use current knee angles as initial guess
        q0 = self.q.copy()
        q0[0] = motor_A_position
        q0[2] = motor_B_position
        
        pin.forwardKinematics(self.model, self.data, q0)
        
        # Solver parameters
        rho = 1e-10
        mu = 1e-4
        eps = 1e-10
        N = 100
        
        q = q0.copy()
        y = np.ones(self.constraint_dim)
        
        self.data.M = np.eye(self.model.nv) * rho
        kkt_constraint = pin.ContactCholeskyDecomposition(self.model, [self.constraint_model])
        
        converged = False
        for k in range(N):
            pin.computeJointJacobians(self.model, self.data, q)
            kkt_constraint.compute(self.model, self.data, [self.constraint_model], [self.constraint_data], mu)
            
            constraint_value = self.constraint_data.c1Mc2.translation
            
            J = pin.getFrameJacobian(
                self.model,
                self.data,
                self.constraint_model.joint1_id,
                self.constraint_model.joint1_placement,
                self.constraint_model.reference_frame,
            )[:3, :]
            
            primal_feas = np.linalg.norm(constraint_value, np.inf)
            dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
            if primal_feas < eps and dual_feas < eps:
                converged = True
                break
            
            # Only update knee angles (indices 1 and 3), keep motor angles fixed
            rhs = np.concatenate([-constraint_value - y * mu, np.zeros(self.model.nv)])
            dz = kkt_constraint.solve(rhs)
            dy = dz[:self.constraint_dim]
            dq = dz[self.constraint_dim:]
            
            # Zero out motor angle updates to keep them fixed
            dq[0] = 0.0
            dq[2] = 0.0
            
            alpha = 1.0
            q = pin.integrate(self.model, q, -alpha * dq)

            # Restore motor angles after integration
            q[0] = motor_A_position
            q[2] = motor_B_position
            y -= alpha * (-dy + y)
        
        if converged:
            # Compute velocity and acceleration from position change if dt provided
            if dt is not None and dt > 0:
                # Store previous velocity for acceleration computation
                v_prev = self.v.copy()
                
                # Use Pinocchio's difference to get velocity in tangent space
                v_raw = pin.difference(self.model, q_prev, q) / dt
                
                # Apply exponential smoothing to velocity
                self.v = self.smoothing * self.v + (1 - self.smoothing) * v_raw
                
                # Compute acceleration from velocity change with smoothing
                a_raw = (self.v - v_prev) / dt
                self.a = self.smoothing * self.a + (1 - self.smoothing) * a_raw
            
            self.q = q.copy()
            
        return converged
    
    def init_visualizer(self, open_browser: bool = True) -> bool:
        """
        Initialize the Meshcat visualizer.
        
        Args:
            open_browser: Whether to open the browser automatically
            
        Returns:
            True if visualization initialized successfully
        """
        try:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            self.viz.initViewer(open=open_browser)
            self.viz.loadViewerModel()
            self.viz.display(self.q)
            return True
        except ImportError as error:
            print(f"Visualization error: {error}")
            return False
    
    def display(self):
        """Update the visualization with the current configuration."""
        if self.viz is not None:
            self.viz.display(self.q)

    def get_motor_torques(self) -> np.ndarray:
        """
        Compute motor torques for the closed-chain mechanism using inverse dynamics.
        
        For a closed-chain mechanism, the equation of motion is:
            M(q)*a + C(q,v)*v + g(q) = tau + J_c^T * lambda
        
        where:
            - M(q) is the mass/inertia matrix
            - C(q,v)*v represents Coriolis and centrifugal forces
            - g(q) is gravity
            - J_c is the constraint Jacobian
            - lambda are the constraint forces
        
        Since passive joints (knees) have zero actuation torque, we use the passive
        joint equations to solve for constraint forces lambda, then substitute back
        to find the motor torques.
            
        Returns:
            Array [tau_motor_A, tau_motor_B] - torques required at the motors
        """
        # Partition indices
        # Motors (indices 0, 2) are actuated, knees (indices 1, 3) are passive
        actuated_idx = [0, 2]  # motor_A, motor_B
        passive_idx = [1, 3]   # knee_A, knee_B
        
        # Compute jacobians
        pin.computeJointJacobians(self.model, self.data, self.q)
        
        # Get the full constraint Jacobian
        J_joint1 = pin.getFrameJacobian(
            self.model,
            self.data,
            self.constraint_model.joint1_id,  # knee_B
            self.constraint_model.joint1_placement,
            self.constraint_model.reference_frame,
        )[:3, :]  # Only position constraint (3D)
        
        J_joint2 = pin.getFrameJacobian(
            self.model,
            self.data,
            self.constraint_model.joint2_id,  # knee_A
            self.constraint_model.joint2_placement,
            self.constraint_model.reference_frame,
        )[:3, :]  # Only position constraint (3D)
        
        # Full constraint Jacobian
        J_constraint = J_joint1 - J_joint2
        
        # Partition constraint Jacobian
        J_c_passive = J_constraint[:, passive_idx]
        J_c_actuated = J_constraint[:, actuated_idx]
        
        # Compute bias forces (gravity + Coriolis) in one RNEA call
        # tau_bias = C(q,v)*v + g(q)
        tau_bias = pin.rnea(self.model, self.data, self.q, self.v, self.a)
        
        # Partition into passive and actuated components
        tau_bias_passive = tau_bias[passive_idx]
        tau_bias_actuated = tau_bias[actuated_idx]
        
        # For passive joints, tau_passive = 0, so:
        #   0 = tau_bias_passive - J_c_passive^T * lambda
        #   lambda = (J_c_passive^T)^{-1} * tau_bias_passive
        lambda_constraint = np.linalg.lstsq(J_c_passive.T, tau_bias_passive, rcond=None)[0]
        
        # Motor torques to counteract bias forces:
        #   tau_motors = tau_actuated - J_c_actuated^T * lambda
        tau_motors = tau_bias_actuated - J_c_actuated.T @ lambda_constraint
        
        # tau_motors = tau_gravity_actuated
        # Store current torques for next iteration
        self.prev_tau_motors = tau_motors.copy()
        
        return tau_motors


VIZ = False

# create the arm
arm = ParallelLegArm()

# initialize visualization
if not arm.init_visualizer(open_browser=VIZ): sys.exit(1)
if VIZ: arm.display()

# move motors
data = np.loadtxt('data/horizontal_out.csv', delimiter=',', encoding='utf-8-sig', skiprows=4)
torques = []
angles_a = data[:, 1]
angles_b = -data[:, 2]

arm.update(angles_a[0], angles_b[0])
for i in range(1, data.shape[0]):
    if arm.update(angles_a[i], angles_b[i], dt=data[i, 0] - data[i-1, 0]):
        if VIZ:
            arm.display()
    else:
        print(f"IK failed at t={data[i, 0]:.2f}")
    torques.append(arm.get_motor_torques())
torques = np.array(torques)

plt.figure(figsize=(10, 6))
plt.plot(data[1:, 0], torques[:, 1], label='Motor A Sim Torque')
plt.plot(data[1:, 0], torques[:, 0], label='Motor B Sim Torque')
# plt.plot(data[1:, 0], data[1:, 1], label='Motor A Angle')
# plt.plot(data[1:, 0], data[1:, 2], label='Motor B Angle')
plt.plot(data[1:, 0], data[1:, 3], label='Motor A Real Torque')
plt.plot(data[1:, 0], data[1:, 4], label='Motor B Real Torque')

plt.plot()

plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Motor Torques Over Time')
plt.legend()
plt.grid(True)
plt.show()