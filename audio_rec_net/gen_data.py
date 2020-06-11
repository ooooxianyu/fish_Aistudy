import os

# 制作标签txt 文件 + 类别
f = open('train_total_data.txt','w')
path = 'D:/AIstudyCode/data/audio_data/speech_commands_train_set_v0.02'
audio_file = os.listdir(path)
print("len", len(audio_file))
label = 0
for file in audio_file:
    img_path = os.listdir(os.path.join(path, file))
    for img  in img_path:
        new_context = os.path.join(file, img) + " " + str(label) +  '\n'
        print(new_context)
        f.write(new_context)
    label = label + 1

f.close()