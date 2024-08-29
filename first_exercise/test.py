result = []

with open("datasets/data_800v/env_vital_signals.txt", "r") as file:
    for line in file:
        data = line.strip().split(",")
        id = str(data[0])
        qpa = str(data[3])
        pulse = str(data[4])
        respiratory_frequency = str(data[5])
        gravity_class = str(data[7])

        result.append(id + "," + qpa + "," + pulse + "," + respiratory_frequency + "," + gravity_class + "\n")
    
with open("data_800v.csv", "w") as file:
    for line in result:
        file.write(line)