import os
import numpy as np
import pandas as pd
from IPython import embed

f_name_old = "vldb_transport"
f_name_new = "modify_"+f_name_old

def Create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def main():
    
    Sensor_map = {"G":0, "A":1, "M":2}

    Dir1 = f_name_old
    Create_directory("modify_"+Dir1)

    for f1_name in os.listdir(Dir1): # Train and Test
        if f1_name != ".DS_Store":
            print(f1_name + "   --1")
            
            Dir2 = os.path.join(Dir1, f1_name)
            Create_directory("modify_"+Dir2)
            for f2_name in os.listdir(Dir2): # Every mode
                if f2_name != ".DS_Store":
                    print(f2_name + "   --2")

                    Dir3 = os.path.join(Dir2, f2_name)
                    Create_directory("modify_"+Dir3)
                    for f3_name in os.listdir(Dir3): # Every status
                        if f3_name != ".DS_Store":
                            print(f3_name + "   --3")

                            Dir4 = os.path.join(Dir3, f3_name)
                            Create_directory("modify_"+Dir4)
                            for f4_name in os.listdir(Dir4): # Every status
                                if f4_name != ".DS_Store":
                                    print(f4_name + "   --4")
                                    # if os.path.isdir(os.path.join(Dir4, f4_name)):
                                    #     print(os.path.join(Dir4, f4_name))
                                    # continue
                                    if not os.path.exists(os.path.join("modify_"+Dir4, f4_name)) and not os.path.isdir(os.path.join(Dir4, f4_name)):
                                        with open(os.path.join(Dir4, f4_name)) as f_r, open(os.path.join("modify_"+Dir4, f4_name), "w") as f_w:
                                            
                                            tmp = {} # [Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz]
                                            a = f_r.read().split("\n")
                                            a = [" ".join(line.split()[-5:]) for line in a if " G: "in line or " A: "in line or " M: "in line]
                                            
                                            for line in a:
                                                time_stamp = line.split()[0]
                                                sensor = line.split()[1].strip(":")
                                                info = " ".join(line.split()[2:])
                                                if time_stamp not in tmp.keys():
                                                    tmp[time_stamp] = ["NAN NAN NAN"]*3
                                                pos = Sensor_map[sensor] # 0, 1, 2
                                                tmp[time_stamp][pos] = info
                                            
                                            for time, all_info in tmp.items():
                                                f_w.write(time + " " + " ".join(all_info) +"\n")

if __name__ == '__main__':
    main()