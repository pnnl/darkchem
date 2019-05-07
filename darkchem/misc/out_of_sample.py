import numpy as np

if __name__ == '__main__':
    insil = {}
    insil['smiles'] = np.load('data/hmdb+unpd_smiles.npy')
    insil['labels'] = np.load('data/hmdb+unpd_labels.npy')

    exp = {}
    exp['smiles'] = np.load('data/vset_smiles.npy')
    exp['labels'] = np.load('data/vset_labels.npy')

    unique = {'smiles': [],
              'labels': []}
    for i in range(len(insil['smiles'])):
        present = False
        print(i)
        for j in range(len(exp['smiles'])):
            if np.array_equal(insil['smiles'][i], exp['smiles'][j]):
                present = True
                break
        if present is False:
            unique['smiles'].append(insil['smiles'][i])
            unique['labels'].append(insil['labels'][i])

    # cast as array
    unique['smiles'] = np.array(unique['smiles'])
    unique['labels'] = np.array(unique['labels'])

    # shuffle
    idx = np.arange(len(unique['smiles']))
    np.random.shuffle(idx)
    unique['smiles'] = unique['smiles'][idx]
    unique['labels'] = unique['labels'][idx]

    # save
    np.save('data/hmdb+unpd-vset_smiles.npy', unique['smiles'])
    np.save('data/hmdb+unpd-vset_labels.npy', unique['labels'])
