## Torque/Angle -> Tip Force Conversion

# converted by evan butler - OSU RAD LAB

import numpy as np
import pandas as pd
import os
from decimal import Decimal as dec
from matplotlib import pyplot as plt

def torque_to_tipfxy(positions,torques,armtype):
    assert armtype in ['osuTraveler'], "Unsupported arm type - calibration not validated"

    def fmodf_0_2pi(x):
        x = np.float32(x)
        twopi = np.float32(2*np.pi)
        r = np.fmod(x, twopi)
        if r < 0:
            r += twopi
        return np.float32(r)

    def fmodf_0_pi(x):
        x = np.float32(x)
        pi = np.float32(np.pi)
        r = np.fmod(x, pi)
        if r < 0:
            r += pi
        return np.float32(r)


    #positions = np.array(positions,dtype=np.float32)
    #torques = np.array(torques,dtype=np.float32)

    position0 = positions[0]
    position1 = positions[1]
    u0 = torques[0]#*-1 # assumed to be the same in low level / csv logs
    u1 = torques[1]

    if armtype == 'osuTraveler':

        # OSU TRAVELER ARM PARAMETERS
        M0_offset = 1.5519 # radians - pulled from traveler low proxy
        M1_offset = 1.5896

        L1 = .1 # meters
        L2 = .2 # meters
        leg_extension_angle = 25.964 # angle of leg extension (not really needed)
        L3 = .12791 # meters (length of leg extension) (not really needed)
        
        mot_pos0 = fmodf_0_2pi(-M0_offset + position0)
        mot_pos1 = fmodf_0_2pi(M1_offset - position1)
        r = mot_pos0-mot_pos1
        diffAng = fmodf_0_pi(0.5*abs(r))
        meanAng = fmodf_0_2pi((mot_pos0 + mot_pos1)/2)
        #meanAng = fmodf_0_2pi(mot_pos1 + diffAng)
        if (mot_pos0 < mot_pos1):
            meanAng -= np.pi
            diffAng = np.pi-diffAng
        l1proj = L1*np.sin(diffAng)
        leglength = np.sqrt(L2*L2 - l1proj*l1proj)+L1*np.cos(diffAng)

        Iq_setpoint0 = u0/0.055
        Iq_setpoint1 = u1/0.055
        u0_adjust = Iq_setpoint0*0.055
        u1_adjust = Iq_setpoint1*0.055

        dummy = -2 / (2*l1proj + L1 * L1 * np.sin(2*diffAng)/np.sqrt(L2*L2-l1proj*l1proj))

        ur = dummy*(u0_adjust - u1_adjust)
        uth = u0_adjust + u1_adjust

        cosMean = np.cos(meanAng)
        sinMean = np.sin(meanAng)

        forceX = (ur*sinMean+uth*cosMean/leglength)
        forceY = -(-ur*cosMean+uth*sinMean/leglength)


        return [forceX,forceY]
    else:
        return [0,0]
    
def normalize(vx, vy, eps=1e-6):
    mag = np.sqrt(vx*vx + vy*vy) + eps
    return vx/mag, vy/mag

def main():

    curve_data = []
    filenames = []
    t = []

    normerrorx = []
    normerrory = []
    angle = []

    pred_fx = []
    pred_fy = []
    real_fx = []
    real_fy = []
    toe_x = []
    toe_y = []


    #print(os.listdir("data/raw"))
    for filename in os.listdir("data/raw/testing"):
            df = pd.read_csv(f"data/raw/testing/{filename}", skiprows=2)
            df = df[['toe_position_x','toe_position_y','toeforce_x','toeforce_y', 'position','position1','torque','torque1']] # takes the 4 i need for tau to fxy (and the 2 sanity checking)
            df.columns = ["xpos","ypos","forcex", "forcey","theta0","theta1","tau0","tau1"] # rename columns
            curve_data.append(df)
            filenames.append(filename)

    # print(df)

    start_idx = 50
    num_samples = 500

    for i in range(num_samples):
        sample = start_idx + i
        t.append(i)
        # theta0i = df["theta0"][sample]
        # print(f"Theta0 Sample {i}: {theta0i}")
      
        theta0i = curve_data[0].iloc[sample]["theta0"]
        theta1i = curve_data[0].iloc[sample]["theta1"]
        tau0i = curve_data[0].iloc[sample]["tau0"]
        tau1i = curve_data[0].iloc[sample]["tau1"]
        realforcex = curve_data[0].iloc[sample]["forcex"]
        realforcey = curve_data[0].iloc[sample]["forcey"]
        forcex, forcey = torque_to_tipfxy([theta0i, theta1i],[tau0i, tau1i],'osuTraveler')


        fxn, fyn = normalize(forcex, forcey)
        rfxn, rfyn = normalize(realforcex, realforcey)

        pred_fx.append(fxn)
        pred_fy.append(fyn)
        real_fx.append(rfxn)
        real_fy.append(rfyn)

        toe_x.append(curve_data[0].iloc[sample]["xpos"])
        toe_y.append(curve_data[0].iloc[sample]["ypos"])

        print(f"Sample {i}:")
        print(f" Predicted Force X: {forcex}, Real Force X: {realforcex}")
        print(f" Predicted Force Y: {forcey}, Real Force Y: {realforcey}")
        print("-----")

        normerrorx.append(np.mean(abs(forcex - realforcex))*100 if realforcex !=0 else 0)
        normerrory.append(np.mean(abs(forcey - realforcey))*100 if realforcey !=0 else 0)
        angle.append(theta0i)

    print(f"Mean Percent Error X: {np.mean(normerrorx)}")
    print(f"Mean Percent Error Y: {np.mean(normerrory)}")

    print("goodbye cruel world!")
    # # Example usage
    # positions = [1.0, 1.0]  # Example joint positions in radians
    # torques = [0.5, 0.5]    # Example joint torques in Nm
    # armtype = 'osuTraveler'

    # tip_forces = torque_to_tipfxy(positions, torques, armtype)
    # print("Tip Forces (X, Y):", tip_forces)

    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(t, normerrorx, label='percent error x forces')
    ax1.plot(t, normerrory, label='percent error y forces')
    
    ax2.plot(t,angle, label='theta0 (rad)', color='green', alpha=0.5)
    plt.xlabel('Sample Index')
    #ax1.label('percent error')
    ax1.set_label('percent error')
    plt.title('percent error in force prediction')
    ax1.legend()
    plt.grid(True)
    plt.show()

#     plt.figure(figsize=(7,7))

# # real force vectors (green)
#     plt.quiver(
#         toe_x, toe_y,
#         real_fx, real_fy,
#         color='g',
#         scale=20,
#         width=0.003,
#         label='Measured'
#     )

# # predicted force vectors (red)
#     plt.quiver(
#         toe_x, toe_y,
#         pred_fx, pred_fy,
#         color='r',
#         scale=20,
#         width=0.003,
#         label='Predicted'
#     )

#     plt.axis('equal')
#     plt.legend()
#     plt.title("Normalized Force Vectors (Direction Only)")
#     plt.xlabel("Toe X")
#     plt.ylabel("Toe Y")
#     plt.grid(True)
#     plt.show()

    angle_err = []

    for fxp, fyp, fxr, fyr in zip(pred_fx, pred_fy, real_fx, real_fy):
        dot = np.clip(fxp*fxr + fyp*fyr, -1.0, 1.0)
        angle_err.append(np.degrees(np.arccos(dot)))

    angle_err = np.array(angle_err)

    err_fx = np.array(real_fx) - np.array(pred_fx)
    err_fy = np.array(real_fy) - np.array(pred_fy)


    plt.figure(figsize=(7,7))
    q = plt.quiver(
        toe_x, toe_y,
        pred_fx, pred_fy,
        angle_err,
        cmap='inferno',
        scale=20,
        width=0.003
    )

    plt.colorbar(q, label='Direction Error (deg)')
    plt.axis('equal')
    plt.title("Predicted Force Direction Colored by Error")
    plt.xlabel("Toe X")
    plt.ylabel("Toe Y")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7,7))

    # predicted
    plt.quiver(toe_x, toe_y, pred_fx, pred_fy,
            color='r', alpha=0.4, scale=20, width=0.003)

    # error vectors
    plt.quiver(toe_x, toe_y, err_fx, err_fy,
            color='k', alpha=0.6, scale=20, width=0.002)

    plt.axis('equal')
    plt.title("Prediction Error Vectors (Black)")
    plt.grid(True)
    plt.show()

    theta0 = np.array(angle)

    plt.figure(figsize=(7,7))
    q = plt.quiver(
        toe_x, toe_y,
        pred_fx, pred_fy,
        theta0,
        cmap='twilight',
        scale=20
    )
    plt.colorbar(q, label='θ₀ (rad)')
    plt.axis('equal')
    plt.title("Predicted Force Colored by Joint Angle")
    plt.grid(True)
    plt.show()


    plt.figure()
    plt.hist(angle_err, bins=50)
    plt.xlabel("Angle Error (deg)")
    plt.ylabel("Count")
    plt.title("Force Direction Error")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()