import argparse
import yaml


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {'cornell': data_dir.joinpath('cornell'),
             'ubuntu': data_dir.joinpath('ubuntu'),
             'dstc6': data_dir.joinpath('dstc6'),
             'test': data_dir.joinpath('test')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}
username = Path.home().name
save_dir = Path('/home/acha21/codes/0.reference_codes/VHCR/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                if key == 'rnncell':
                    value = rnncell_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        # Pickled Dataframes
        # for example
        # self.sentences_path = self.data_dir.joinpath('sentences.pkl')
        # self.sentence_length_path = self.data_dir.joinpath('sentence_length.pkl')
        # self.conversation_length_path = self.data_dir.joinpath('conversation_length.pkl')

        # Save path
        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = save_dir.joinpath(self.data, self.model, time_now)
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str
        
def load_yaml(filepath):
    """Load configure parameters given file path

    Args:
        filepath (str): yaml file path

    Returns:
        dict: dictionary that contains the parameters

    """
    with open(filepath, "r") as f:
        data = yaml.load(f)

    for k in data:
        if data[k] == "None":
            data[k] = None

    return data


def save_yaml(filepath, data):
    """ 
    Args:
        filepath (str): 
        data: 

    Returns:
        None
    """
    with open(filepath, "w") as f:
        yaml.dump(data, f)
        
def get_config(parse=True, **optional_kwargs):
    conf_parser = argparse.ArgumentParser(add_help=False)
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    # Train
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--eval_batch_size', type=int, default=80)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Model
    parser.add_argument('--model', type=str, default='?',
                        help='one of {?, ?, etc}')

    parser.add_argument('--rnn', type=str, default='gru')

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)
    
    # Data
    parser.add_argument('--data', type=str, default='ubuntu')
    
    # Parse arguments
    if parse:
        kwargs = parser.parse_args(remaining_argv)
    else:
        kwargs, un_parsed = parser.parse_known_args(remaining_argv)
    # Namespace => Dictionary
    kwargs = vars(kwargs)

    if cnf_args.config:
        # if yaml file is given it takes precedence over parameters given in command line
        kwargs.update(yaml_config_dic)
        # User warning!
        yaml_param_names = yaml_config_dic.keys()
        param_name_in_cmd = set([remaining_argv[i][2:] for i in range(0, len(remaining_argv), 2)])
        repeated_params = yaml_param_names & param_name_in_cmd

        if len(repeated_params):
            print(f"{repeated_params} are repeated")
            print("We follow yaml setting!!")
            for k in repeated_params:
                print(f"   {k}:{yaml_config_dic[k]} in {cnf_args.config}")

        all_default_keys = kwargs.keys()
        not_defined_params_in_yaml = all_default_keys - yaml_param_names
        if len(not_defined_params_in_yaml) > 0:
            print(f"{not_defined_params_in_yaml} are parameter set by default")

    # the parameters hard-coded takes the precedence over all the other parameters.
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
    