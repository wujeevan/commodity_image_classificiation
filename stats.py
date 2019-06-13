from functions import Functions
import pandas as pd
import matplotlib.pyplot as plt
fct = Functions()

dirs = '../data/daily-usage'
x, y, n, y_dict = fct.read_images(dirs, target_size=(64,64))
# m = [(y_dict[i], val) for i, val in n.items()]
# pd = pd.DataFrame(m)
# pd.to_excel('stats.xlsx')

row, col = 8, 10
per = 2
cnt = {}
now = 1
plt.figure()
for i, image in enumerate(x):
    if y[i] not in cnt:
        cnt[y[i]] = 0
    cnt[y[i]] += 1
    if cnt[y[i]] <= per:
        plt.subplot(row, col, now)
        plt.imshow(image)
        plt.axis('off')
        now = now + 1
    else:
        continue

