# -*- coding: utf-8 -*-
import argparse
import importlib
import sys

def main():
    parser = argparse.ArgumentParser(description='Robotics final project')
    parser.add_argument('task', help='task id, choose in (1, 2, 3, 4) (only number)')
    args = parser.parse_args()

    task_id = 0

    if args.task in ('1', 't1', 'task1'):
        task_id = 1
    elif args.task in ('2', 't2', 'task2'):
        task_id = 2
    elif args.task in ('3', 't3', 'task3'):
        task_id = 3
    elif args.task in ('4', 't4', 'task4'):
        task_id = 4
    else:
        print('Wrong format input: ' + str(args.task))
        print('Plz try again, This process will be exited.')
        sys.exit(1)

    print('Run task ' + str(task_id))
    task_module = importlib.import_module('t' + str(task_id))
    task_module.main()

if __name__ == '__main__':
    main()
