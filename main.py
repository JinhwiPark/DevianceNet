from functions_deviance import *
from sklearn.metrics import accuracy_score
import random
import warnings
from tqdm import tqdm
from config import args as args_config
import numpy as np
import json
import torchvision.transforms as transforms

warnings.filterwarnings(action='ignore')


def train(model, device, train_loader, optimizer, epoch, args):
    # set model as training mode
    model.train()

    train_total_loss = 0
    train_detection_loss = 0
    train_classification_loss = 0

    SEA_epoch_accuracy = 0
    DIA_epoch_accuracy = 0

    N_count = 0  # counting total trained sample in one epoch

    tbar = tqdm(train_loader)

    for batch_idx, (X, y) in enumerate(tbar):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        output_classification, output_detection = model(X)  # output size = (batch, number of classes)
        DEloss, CLloss = H_loss(output_classification, output_detection, y, device, args)
        Totalloss = DEloss + CLloss

        # to compute accuracy
        y_pred = torch.max(output_classification, 1)[1]  # y_pred != output
        y_pred_de = torch.max(output_detection, 1)[1]

        y_detection = torch.ones(y.size(0)).to(device)
        y_detection[y != 4] = 0

        step_score_classification = accuracy_score(y.cpu(), y_pred.cpu())  # classification
        step_score_detection = accuracy_score(y_detection.cpu(), y_pred_de.cpu())  # detection

        DIA_epoch_accuracy += step_score_detection
        SEA_epoch_accuracy += step_score_classification
        train_total_loss += Totalloss.item()
        train_detection_loss += DEloss.item()
        train_classification_loss += CLloss.item()

        optimizer.zero_grad()
        Totalloss.backward()
        optimizer.step()

        # show information
        log_interval = int(len(train_loader) / 4)
        if log_interval != 0 and (batch_idx + 1) % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Totaloss:{:.4f} DEloss:{:.4f} CLloss:{:.4f} | CL_Accu:{:.2f}% DE_Accu:{:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    Totalloss.item(), DEloss.item(), CLloss.item(),
                    100 * step_score_classification, 100 * step_score_detection))

    train_total_loss /= len(train_loader)
    train_detection_loss /= len(train_loader)
    train_classification_loss /= len(train_loader)
    SEA_epoch_accuracy /= len(train_loader)
    DIA_epoch_accuracy /= len(train_loader)

    return train_total_loss, train_detection_loss, train_classification_loss, DIA_epoch_accuracy, SEA_epoch_accuracy


def val(model, device, optimizer, SEA_val_loader, DIA_val_loader, epoch, args):
    model.eval()

    global score_min_CL
    global score_min_DE

    val_total_loss = 0
    val_detection_loss = 0
    val_classification_loss = 0

    SEA_GT = []
    SEA_pred = []
    DIA_GT = []
    DIA_pred = []

    with torch.no_grad():
        print(" SEA Evaluation")
        for X, y in tqdm(SEA_val_loader):  # batch 별로 분할된거임
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            output_classification, output_detection = model(X)  # output size = (batch, number of classes)
            DEloss, CLloss = H_loss(output_classification, output_detection, y, device, args)
            Totalloss = DEloss + CLloss

            val_total_loss += Totalloss.item()
            val_detection_loss += DEloss.item()
            val_classification_loss += CLloss.item()

            # to compute accuracy
            y_pred = torch.max(output_classification, 1)[1]  # y_pred != output

            SEA_GT.extend(y)
            SEA_pred.extend(y_pred)
        print(" DIA Evaluation")
        for X, y in tqdm(DIA_val_loader):  # batch 별로 분할된거임
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            output_classification, output_detection = model(X)  # output size = (batch, number of classes)
            DEloss, CLloss = H_loss(output_classification, output_detection, y, device, args)
            Totalloss = DEloss + CLloss

            val_total_loss += Totalloss.item()
            val_detection_loss += DEloss.item()
            val_classification_loss += CLloss.item()

            # to compute accuracy
            y_pred_de = torch.max(output_detection, 1)[1]
            y_detection = torch.ones(y.size(0)).to(device)
            y_detection[y != 4] = 0

            DIA_GT.extend(y_detection)
            DIA_pred.extend(y_pred_de)

    val_total_loss /= (len(SEA_val_loader) + len(DIA_val_loader))
    val_detection_loss /= (len(SEA_val_loader) + len(DIA_val_loader))
    val_classification_loss /= (len(SEA_val_loader) + len(DIA_val_loader))

    # compute accuracy
    SEA_GT = torch.stack(SEA_GT, dim=0).cpu().data.squeeze().numpy()
    SEA_pred = torch.stack(SEA_pred, dim=0).cpu().data.squeeze().numpy()
    DIA_GT = torch.stack(DIA_GT, dim=0).cpu().data.squeeze().numpy()
    DIA_pred = torch.stack(DIA_pred, dim=0).cpu().data.squeeze().numpy()

    classification_score = accuracy_score(SEA_GT, SEA_pred)
    detection_score = accuracy_score(DIA_GT, DIA_pred)
    MAE = float(sum(abs(SEA_GT - SEA_pred))) / len(SEA_GT)

    print("============= Validation =============")
    print(
        'Validation set ({:d} samples) >>  CLloss:{:.4f} DEloss:{:.4f} Totalloss:{:.4f}|| SEA: {:.2f}%  DIA: {:.2f}%'.
            format(len(SEA_GT), val_classification_loss, val_detection_loss, val_total_loss,
                   100 * classification_score, 100 * detection_score))
    print('MAE = ', MAE)

    if classification_score > score_min_CL:
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'cnn_encoder_epoch{}.pth'.format(
            epoch + 1)))  # save spatial_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(args.save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))
        print("Update Best Accuracy for SEA")
        print(" SEA : {} -> {}".format(score_min_CL, classification_score))
        score_min_CL = classification_score
    elif detection_score > score_min_DE:
        torch.save(model.state_dict(), os.path.join(args.save_model_path, 'cnn_encoder_epoch{}.pth'.format(
            epoch + 1)))  # save spatial_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(args.save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))
        print("Update Best Accuracy for  DIA")
        print(" DIA : {} -> {}".format(score_min_DE, detection_score))
        score_min_DE = detection_score

    return classification_score, detection_score, val_total_loss, val_detection_loss, val_classification_loss, MAE

def test(model, device, test_loader, test_metric_type):
    # set model as testing mode
    model.eval()

    gt = []
    prediction = []

    with torch.no_grad():
        print("\n{} Evaluation".format(test_metric_type))
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device).view(-1, )
            output_classification, output_detection = model(X)

            # to compute accuracy
            if test_metric_type == 'SEA':
                y_pred = torch.max(output_classification, 1)[1]
                y_gt = y
            elif test_metric_type == 'DIA':
                y_pred = torch.max(output_detection, 1)[1]
                y_gt = torch.ones(y.size(0)).to(device)
                y_gt[y != 4] = 0
            gt.extend(y_gt)
            prediction.extend(y_pred)

    gt = torch.stack(gt, dim=0).cpu().data.squeeze().numpy()
    prediction = torch.stack(prediction, dim=0).cpu().data.squeeze().numpy()
    test_score = accuracy_score(gt, prediction)

    print('Test set ({:d} samples) >> {}: {:.2f}%'.format(len(gt), test_metric_type, 100 * test_score))
    if test_metric_type == 'SEA':
        MAE = float(sum(abs(gt - prediction))) / len(gt)
        print('MAE = ', MAE)


if __name__ == '__main__':

    torch.manual_seed(args_config.seed)
    np.random.seed(args_config.seed)
    random.seed(args_config.seed)
    torch.cuda.manual_seed(args_config.seed)
    torch.cuda.manual_seed_all(args_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    score_min_CL = args_config.score_min_CL
    score_min_DE = args_config.score_min_DE

    if not args_config.test_only:
        backup_source_code(args_config.output_path + '/code')
        if not os.path.exists(args_config.output_path):
            os.makedirs(args_config.output_path)
        if not os.path.exists(args_config.save_model_path):
            os.makedirs(args_config.save_model_path)
        with open(args_config.output_path + '/args.json', 'w') as args_json:
            json.dump(args_config.__dict__, args_json, indent=4)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_config)):
        print(key, ':', getattr(args_config, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    # Select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 0, args_config.frame_num, 1
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # transform
    transform = transforms.Compose([transforms.CenterCrop((480, 640)),
                                    transforms.Resize([args_config.img_x, args_config.img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    if use_cuda and not args_config.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    params = {'batch_size': args_config.batch_size, 'shuffle': True, 'num_workers': args_config.num_threads,
              'pin_memory': True} if use_cuda else {}
    params_test = {'batch_size': args_config.batch_size, 'shuffle': False, 'num_workers': args_config.num_threads,
                   'pin_memory': True} if use_cuda else {}
    if args_config.model == 'DevianceNet':
        from models.deviancenet import *

        print("DevianceNet is loaded!!")
        net = DevianceNet(args_config, device=device).to(device)
    elif args_config.model == 'hatnet':
        from models.hatnet import *

        net = HATNet(classifier_type=args_config.classifier_type, drop_p=args_config.dropout).to(device)
    elif args_config.model == 'hatnet_SP':
        from models.hatnet_SP import *

        net = HATNet_SP(SPtype=args_config.superpoint_type, classifier_type=args_config.classifier_type,
                        device=device, drop_p=args_config.dropout).to(device)

    if not args_config.weight_load_pth is None:
        net.load_state_dict(load_pth(args_config.weight_load_pth), strict=False)

    # Freeze supernet
    if args_config.superpoint_freeze and args_config.superpoint_type != None:
        for name, param in net.named_parameters():
            if name.startswith('supernet'):
                param.requires_grad = False

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    if args_config.korea_total:
        cities = ['Busan', 'Daegu', 'Daejeon', 'Incheon', 'Seoul']

        args_config.train_folder_directory = [
            os.path.join(args_config.train_folder_directory, '{}_train_SEA'.format(city, args_config.classifier_type))
            for city in cities]

        args_config.SEA_folder_directory = [os.path.join(args_config.SEA_folder_directory, '{}_test_SEA'.format(city))
                                            for city in cities]
        args_config.DIA_folder_directory = [os.path.join(args_config.DIA_folder_directory, '{}_test_DIA'.format(city))
                                            for city in cities]

    if not args_config.test_only:
        print("Dataset Directories")
        print('Directory for train : ', args_config.train_folder_directory)
        print('Directory for SEA evaluation : ', args_config.SEA_folder_directory)
        print('Directory for DIA evaluation : ', args_config.DIA_folder_directory)

        if not os.path.exists(args_config.output_path):
            os.makedirs(args_config.output_path)
        if not os.path.exists(args_config.save_model_path):
            os.makedirs(args_config.save_model_path)

        if args_config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=args_config.lr)  # optimize all cnn parameters
        elif args_config.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=args_config.lr)  # optimize all cnn parameters

        train_loader = data.DataLoader(
            DatasetDeviance(args_config.train_folder_directory, selected_frames, transform=transform,
                            partition=args_config.partition, direction=args_config.direction), **params)
        SEA_val_loader = data.DataLoader(
            DatasetDeviance(args_config.SEA_folder_directory, selected_frames, transform=transform), **params_test)
        DIA_val_loader = data.DataLoader(
            DatasetDeviance(args_config.DIA_folder_directory, selected_frames, transform=transform), **params_test)

        for epoch in range(args_config.epochs):
            train_losses, DE_losses, CE_losses, train_DEscores, train_CLscores = train(net, device, train_loader,
                                                                                       optimizer, epoch, args_config)
            val_CLaccuracy, val_DEaccuracy, val_loss, val_DE_losses, val_CE_losses, mae = val(net, device,
                                                                                              optimizer,
                                                                                              SEA_val_loader,
                                                                                              DIA_val_loader,
                                                                                              epoch,
                                                                                              args_config)
    else:
        print("TEST ONLY!!")

        print("\nDataset Directories")

        if args_config.test_metric == 'SEA':
            print('Directory for SEA evaluation : ', args_config.SEA_folder_directory)
            test_path = args_config.SEA_folder_directory
            SEA_test_loader = data.DataLoader(DatasetDeviance(test_path, selected_frames, transform=transform),
                                              **params_test)
            test(net, device, SEA_test_loader, args_config.test_metric)
        if args_config.test_metric == 'DIA':
            print('Directory for DIA evaluation : ', args_config.DIA_folder_directory)
            test_path = args_config.DIA_folder_directory
            DIA_test_loader = data.DataLoader(DatasetDeviance(test_path, selected_frames, transform=transform),
                                              **params_test)
            test(net, device, DIA_test_loader, args_config.test_metric)
