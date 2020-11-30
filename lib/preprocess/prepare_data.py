import pandas as pd


def extract_label(data_path, mode, n=None):
    folders = {'train': '1346', 'val': '1345', 'test': '1347'}
    with open(data_path) as f:
        lines = f.readlines()
        f.close()
        images = []
        for line in lines:
            folder = line.split('/')[1]
            if folder == folders[mode]:
                label = line.split('_')[1]
                if len(label) > 12:
                    continue
                label = label.upper()
                file_path = line.split(' ')[0]
                file_path = 'Data' + file_path[1:]
                images.append([file_path, label])
                if n is not None and len(images) >= n:
                    break
        images_df = pd.DataFrame(images, columns=['Image', 'Label'])
        return images_df


datasets = ['train', 'val', 'test']
for dataset in datasets:
    data = extract_label('Data/annotation_train.txt', mode=dataset)
    dest = 'Data/' + dataset + '_data.csv'
    data.to_csv(dest, index=False)
