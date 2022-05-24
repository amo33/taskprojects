import argparse
from kf_utility.load_data_for_one import *
from kf_utility.load_data_c_for_one import *
from kf_utility.resnest import *
# from utility.engine_test_att import *
from kf_utility.engine_test_style_for_one import *
from kf_utility.engine_test_att_for_one import *
from kf_utility.ml_gcn import *
import ray

num_cpus = 16
ray.init(num_cpus = num_cpus, ignore_reinit_error=True)

@ray.remote
def run_attribute_classifier(model_name ='category', args=None):

    use_gpu = torch.cuda.is_available()
    state = {'batch_size': args.batch_size, 'image_size': args.image_size,
             'evaluate': args.evaluate}

    if model_name == 'category':
        num_classes = 21
        #labels = '블라우스'
        state['resume'] = './kf_checkpoint/kfashion_category/model_category_best.pth.tar'
    elif model_name == 'detail':
        num_classes = 40
        #labels = '스터드'
        state['resume'] = './kf_checkpoint/kfashion_detail/model_detail_best.pth.tar'
    elif model_name == 'texture':
        num_classes = 27
        #labels = '패딩'
        state['resume'] = './kf_checkpoint/kfashion_texture/model_texture_best.pth.tar'
    elif model_name == 'print':
        num_classes = 21
        #labels = '페이즐리'
        state['resume'] = './kf_checkpoint/kfashion_print/model_print_best.pth.tar'
    elif model_name == 'style':
        num_classes = 10
        #labels = 'TRADITIONAL'
        state['resume'] = './kf_checkpoint/kfashion_style/model_style_best.pth.tar'

    criterion = nn.MultiLabelSoftMarginLoss()
    state['evaluate'] = True

    if model_name == 'style':
        startTime = time.time()
        
        test_dataset = load_data_for_one(root = args.data, phase='test', input_data_path=args.input_img_path, labels=None, inp_name=args.wordvec)
        model = gcn_resnet101(num_classes=num_classes, t=0.03, adj_file='./kf_data/kfashion_style/custom_adj_final.pkl')
        engine = Engine_style(state)
        a = engine.learning_for_one(model, criterion, test_dataset)
        print(model_name +": print time spent :{:.4f}".format(time.time() - startTime))
    else:
        startTime = time.time()
        test_dataset = load_data_c_for_one(attribute=model_name, phase='test', num_classes=num_classes, input_data_path=args.input_img_path, labels=None)
        model = resnest50d(pretrained=False, nc=num_classes)
        engine = Engine_att(state)
        a = engine.learning_for_one(model, criterion, test_dataset, model_name)
        print(model_name +": print time spent :{:.4f}".format(time.time() - startTime))
    # a = engine.learning(model, criterion, test_dataset, model_name)
    return a

if name == '__main__':
    #startTime = time.time()

    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument('--input_img_path', default='./mill_sample/mill3.png') 
    parser.add_argument('--data', default='./kf_data/kfashion_style') #data==> directory? input_img_path와 같은거 아닌가요?
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('--wordvec', default='./kf_data/kfashion_style/custom_glove_word2vec_final.pkl')
    parser.add_argument('-j', '--workers', default=12, type=int)
    parser.add_argument('--device_ids', default=[0], type=int, nargs='+')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                        help='evaluate model on validation set')
    
    args = parser.parse_args()
    # ('texture', args) , ('print',args),
    options = [('category',args),('detail',args), ('style',args)]
    results_id = [run_attribute_classifier.remote(x,y) for x,y in enumerate(options)]
    results = ray.get(results_id)
    print('input_img_path: ', args.input_img_path)
    '''
    p1 = run_attribute_classifier(model_name='category',args=args)
    print('category: ', p1)
    p2 = run_attribute_classifier(model_name='detail',args=args)
    print('detail: ', p2)
    p3 = run_attribute_classifier(model_name='texture',args=args)
    print('texture: ', p3)
    p4 = run_attribute_classifier(model_name='print',args=args)
    print('print: ', p4)
    p5 = run_attribute_classifier(model_name='style',args=args)
    print('style: ', p5)
    '''
    print(results)



    