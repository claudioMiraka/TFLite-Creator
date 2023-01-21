import os
from modules.utils import *
from modules.downloader import *
from modules.csv_downloader import *

from modules.utils import bcolors as bc

def bounding_boxes_images(args, DEFAULT_OID_DIR):

	if not args.Dataset:
		dataset_dir = os.path.join(DEFAULT_OID_DIR, 'Dataset')
		csv_dir = os.path.join(DEFAULT_OID_DIR, 'csv_folder')
	else:
		dataset_dir = os.path.join(DEFAULT_OID_DIR, args.Dataset)
		csv_dir = os.path.join(DEFAULT_OID_DIR, 'csv_folder')

	name_file_class = 'class-descriptions-boxable.csv'
	CLASSES_CSV = os.path.join(csv_dir, name_file_class)

	if args.command == 'downloader':

		if args.type_csv is None:
			print(bc.FAIL + 'Missing type_csv argument.' + bc.ENDC)
			exit(1)
		if args.classes is None:
			print(bc.FAIL + 'Missing classes argument.' + bc.ENDC)
			exit(1)
		if args.multiclasses is None:
			args.multiclasses = 0

		folder = ['train', 'validation', 'test']
		file_list = ['train-annotations-bbox.csv', 'validation-annotations-bbox.csv', 'test-annotations-bbox.csv']

		if args.classes[0].endswith('.txt'):
			with open(args.classes[0]) as f:
				args.classes = f.readlines()
				args.classes = [x.strip() for x in args.classes]
		else:
				args.classes = [arg.replace('_', ' ') for arg in args.classes]

		if args.multiclasses == '0':

			mkdirs(dataset_dir, csv_dir, args.classes, args.type_csv)

			for classes in args.classes:

				print(bc.INFO + 'Downloading {}.'.format(classes) + bc.ENDC)
				class_name = classes

				error_csv(name_file_class, csv_dir, args.yes)
				df_classes = pd.read_csv(CLASSES_CSV, header=None)

				class_code = df_classes.loc[df_classes[1] == class_name].values[0][0]

				if args.type_csv == 'train':
					name_file = file_list[0]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[0], dataset_dir, class_name, class_code)
					else:
						download(args, df_val, folder[0], dataset_dir, class_name, class_code, threads = int(args.n_threads))

				elif args.type_csv == 'validation':
					name_file = file_list[1]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[1], dataset_dir, class_name, class_code)
					else:
						download(args, df_val, folder[1], dataset_dir, class_name, class_code, threads = int(args.n_threads))

				elif args.type_csv == 'test':
					name_file = file_list[2]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[2], dataset_dir, class_name, class_code)
					else:
						download(args, df_val, folder[2], dataset_dir, class_name, class_code, threads = int(args.n_threads))

				elif args.type_csv == 'all':
					for i in range(3):
						name_file = file_list[i]
						df_val = TTV(csv_dir, name_file, args.yes)
						if not args.n_threads:
							download(args, df_val, folder[i], dataset_dir, class_name, class_code)
						else:
							download(args, df_val, folder[i], dataset_dir, class_name, class_code, threads = int(args.n_threads))
				else:
					print(bc.ERROR + 'csv file not specified' + bc.ENDC)
					exit(1)

		elif args.multiclasses == '1':

			class_list = args.classes
			print(bc.INFO + 'Downloading {} together.'.format(class_list) + bc.ENDC)
			multiclass_name = ['_'.join(class_list)]
			mkdirs(dataset_dir, csv_dir, multiclass_name, args.type_csv)

			error_csv(name_file_class, csv_dir, args.yes)
			df_classes = pd.read_csv(CLASSES_CSV, header=None)

			class_dict = {}
			for class_name in class_list:
				class_dict[class_name] = df_classes.loc[df_classes[1] == class_name].values[0][0]

			for class_name in class_list:

				if args.type_csv == 'train':
					name_file = file_list[0]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[0], dataset_dir, class_name, class_dict[class_name], class_list)
					else:
						download(args, df_val, folder[0], dataset_dir, class_name, class_dict[class_name], class_list, int(args.n_threads))

				elif args.type_csv == 'validation':
					name_file = file_list[1]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[1], dataset_dir, class_name, class_dict[class_name], class_list)
					else:
						download(args, df_val, folder[1], dataset_dir, class_name, class_dict[class_name], class_list, int(args.n_threads))

				elif args.type_csv == 'test':
					name_file = file_list[2]
					df_val = TTV(csv_dir, name_file, args.yes)
					if not args.n_threads:
						download(args, df_val, folder[2], dataset_dir, class_name, class_dict[class_name], class_list)
					else:
						download(args, df_val, folder[2], dataset_dir, class_name, class_dict[class_name], class_list, int(args.n_threads))

				elif args.type_csv == 'all':
					for i in range(3):
						name_file = file_list[i]
						df_val = TTV(csv_dir, name_file, args.yes)
						if not args.n_threads:
							download(args, df_val, folder[i], dataset_dir, class_name, class_dict[class_name], class_list)
						else:
							download(args, df_val, folder[i], dataset_dir, class_name, class_dict[class_name], class_list, int(args.n_threads))
