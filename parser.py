import yaml

with open('config.yml','r') as data:
	try:
		config = yaml.safe_load(data)
	except yaml.YAMLError as exc:
		print(exc)

params = {}

params['train_images_path'] = config['dataset']['train_images_path']
params['train_labels_path'] = config['dataset']['train_labels_path']

params['check_freq'] = config['train']['check_freq']
params['output_weight_path'] = config['train']['output_weight_path']
params['input_weight_path'] = config['train']['input_weight_path']
params['num_epochs'] = config['train']['num_epochs']
params['binarize'] = config['train']['binarize']
params['rgb_weightage'] = config['train']['rgb_weightage']
params['freeze_first_block'] = config['train']['freeze_first_block']
params['pretrained'] = config['train']['pretrained']
params['learning_rate'] = config['train']['learning_rate']
params['momentum'] = config['train']['momentum']
params['weight_decay'] = config['train']['weight_decay']
params['step_size'] = config['train']['step_size']
params['gamma'] =  config['train']['gamma']

params['horizontal_flip'] = config['transform']['horizontal_flips']

params['test_images_path'] = config['infer']['test_images_path']
params['checkpoint_path'] = config['infer']['checkpoint_path']

params['debug_mode'] = config['debug_mode']