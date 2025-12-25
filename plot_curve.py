import datetime
import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate):
    """
    绘制训练损失和学习率曲线
    Args:
        train_loss: 训练损失
        learning_rate: 学习率
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'r-', label='train loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(learning_rate, 'b-', label='learning rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    plt.close()


def plot_map(val_map):
    """
    绘制mAP曲线
    Args:
        val_map: 验证集mAP
    """
    plt.figure(figsize=(10, 5))
    plt.plot(val_map, 'r-', label='val mAP')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.savefig('mAP.png')
    plt.close()


def plot_mrse(val_mrse):
    """
    绘制MRSE曲线
    Args:
        val_mrse: 验证集MRSE
    """
    plt.figure(figsize=(10, 5))
    plt.plot(val_mrse, 'r-', label='val MRSE')
    plt.title('Validation MRSE')
    plt.xlabel('Epoch')
    plt.ylabel('MRSE')
    plt.legend()
    plt.savefig('mrse_per_round.png')
    plt.close()
