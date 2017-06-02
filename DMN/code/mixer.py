import sys
import os

folder = sys.argv[1]

keep = set(str(x) for x in set(range(21)) - set([int(x) for x in sys.argv[2:]]))

files = os.listdir(folder)

files = [file for file in files if any("qa" + num + "_" in file for num in keep)]

if not os.path.exists("tasks"):
    os.mkdir("tasks")

for category in ["train", "valid", "test"]:
    curr = [file for file in files if category in file]
    with open(os.path.join("tasks", category + "_all.txt"), "w") as target:
        for file_name in curr:
            with open(os.path.join(folder, file_name)) as file:
                file_data = file.read()
            target.write(file_data)

            if category == "test":
                with open(os.path.join('tasks', file_name), 'w') as test_target:
                    test_target.write(file_data)
