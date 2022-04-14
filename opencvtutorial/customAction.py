import argparse
import re 
class customAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(customAction, self).__init__(*args, **kwargs)
    def __call__(self, parser,namespace, values, option_string):
        print("class print:",values)
        if ',' in values:
            values = re.split(',', values)
            
        if ' ' in values:
            values= re.split(' ', values)
            
        setattr(namespace, self.dest, values)