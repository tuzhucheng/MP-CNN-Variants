from random import randint, uniform
from subprocess import call

count = 25
for id in range(count):
    learning_rate = round(10 ** uniform(-4, -2), 4)
    eps = round(10 ** uniform(-4, -1), 4)
    reg = round(10 ** uniform(-4, -1), 4)

    filename = "grid_msrvid_lr_{learning_rate}_eps_{eps}_reg_{reg}.txt".format(learning_rate=learning_rate, eps=eps, reg=reg)
    command = "python main.py saved_models/msrvid_grid.castor --epochs 30 --dataset msrvid --batch-size 16 --lr {learning_rate} --epsilon {eps} --regularization {reg}".format(learning_rate=learning_rate, eps=eps, reg=reg)

    print("Running: " + command)
    with open(filename, 'w') as outfile:
        call(command, shell=True, stderr=outfile)
