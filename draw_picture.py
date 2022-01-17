import json
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    with open('logs.json', 'r') as f:
        stat_dict = json.load(f)
        epoch_list = sorted(list(int(x) for x in stat_dict.keys()))
        disc_list = list(float(stat_dict[str(x)]['disc_loss']) for x in epoch_list)
        gen_list = list(float(stat_dict[str(x)]['gen_loss']) for x in epoch_list)
        fid_list = list(float(stat_dict[str(x)]['fid_score']) for x in epoch_list)
    return disc_list, gen_list, fid_list

def draw_picture(data, title):
    plt.ylim(min(data)/1.2, max(data)*1.1)
    X = np.arange(len(data))
    X = X*5
    plt.plot(X, data, 'r')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.show()
    
if __name__ == "__main__":
    disc_list, gen_list, fid_list = get_data()
    draw_picture(disc_list, 'Disc Loss')
    draw_picture(gen_list, 'Gen Loss')
    draw_picture(fid_list, 'FID Score')