%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = 'Times New Roman'

height, width = 4, 5
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
plt.subplots_adjust(hspace=0.3)

ego_keys = ['speed', 'lateral_action', 'has_left', 'has_right']
discretes = [False, True, True, True]

for idx, key in enumerate(ego_keys):
    row, col = idx // ncols, idx % ncols
    if discretes[idx]:
        counter = Counter(ego_data[key])
        if key == 'lateral_action':
            counter = {'keep': counter[0], 'left': counter[1] + counter[3], 'right': counter[2] + counter[4]}
        counter_sum = sum(counter.values())
        for a in counter:
            counter[a] /= counter_sum
        axes[row][col].bar(counter.keys(), counter.values())
        axes[row][col].set_xticks(list(counter.keys()))
        print(key, counter)
    else:
        axes[row][col].hist(ego_data[key], bins=50, density=True)
        counter_sum = len(ego_data[key])
    axes[row][col].set_ylabel('ratio')
    axes[row][col].set_title(f'ego.{key}\ntotal={counter_sum}')
fig.savefig('ego.png', dpi=300)
