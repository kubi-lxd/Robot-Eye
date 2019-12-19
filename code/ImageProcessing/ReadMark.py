import cv2
from Exception import MyException
from enum import Enum

MarkFilePath = '../../data/total/mark total.txt'
FigFolderName = '../../figures/figures/'
ResultImgPath = '../../figures/examples/result/'


class OutputMode(Enum):
    PrintScreen = 1
    FileOutput = 2


class ReadMarkError(MyException):
    def __init__(self):
        super(ReadMarkError, self).__init__(code=1001,
                                            message='Number of mark points does not match the record',
                                            args=('ReadMarkError',))


def readname(filename):
    head, alpha, beta, tail = filename.split('_')
    gama, end = tail.split('.')
    # 转整数
    alpha = int(alpha)
    beta = int(beta)
    gama = int(gama)
    return alpha, beta, gama


def readmark(filename):
    """
    :param filename: 图片名称 典型值'image_178_3_153.bmp'
    :return: 数组list 典型值[[-600, 1800, 1141.3, 298.25], [-600, 1500, 1085.8, 430.25]]
    数组的长度表示标注点的数量 每个标注点包含四个参数 X Y x y 大写为真实坐标 小写为图像坐标
    xy为小数是因为标注时图片太大经过缩放 产生了小数坐标标注 参考小数坐标寻找最临近亮点即可
    """
    alpha, beta, gama = readname(filename)
    with open(MarkFilePath, 'r', encoding='UTF-8') as f:
        result = []
        length = 0
        for i, line in enumerate(f):
            X, Y, x, y, alpha_here, beta_here, gama_here, num = line.split('\t')
            X = int(X)
            Y = int(Y)
            x = float(x)
            y = float(y)
            alpha_here = int(alpha_here)
            beta_here = int(beta_here)
            gama_here = int(gama_here)
            num = int(num)
            if alpha_here == alpha and beta_here == beta and gama_here == gama:
                result.append([X, Y, x, y])
                length = num
        if not len(result) == length:
            raise ReadMarkError
        else:
            return result


def change(x, y, xsize, ysize):
    if x <= 0:
        x = 1
    if y <= 0:
        y = 1
    if x >= xsize:
        x = xsize
    if y >= ysize:
        y = ysize
    return x, y


def checkmark(figname, mode=OutputMode.PrintScreen):
    """
    输入图片名称 显示检查标注点 会打开一张图片
    :param mode:
    :param figname:图片名称
    :return:无
    """
    figname = FigFolderName + figname
    fig = cv2.imread(figname)
    xsize = fig.shape[1]
    ysize = fig.shape[0]
    # 拿到标注点信息
    points = readmark(figname)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 标点 画圈 写字
    if len(points):
        for point in points:
            p = change(int(point[2]), int(point[3]), xsize, ysize)
            cv2.circle(fig, (p[0], p[1]), 15, (0, 0, 255), 2)
            p = change(int(point[2]), int(point[3]) - 10, xsize, ysize)
            cv2.putText(fig, str(int(point[0])) + ',' + str(int(point[1])),
                       (p[0], p[1]), font, 1, (255, 255, 255), 1)
    # 显示
    if mode == OutputMode.PrintScreen:
        cv2.namedWindow('checkmark', cv2.WINDOW_AUTOSIZE)
        # 缩放 防止超出屏幕
        figsize = fig.shape
        fig = cv2.resize(fig, (int(figsize[1] * 0.5), int(figsize[0] * 0.5)), cv2.INTER_LINEAR)
        cv2.imshow('see_image', fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == OutputMode.FileOutput:
        print('saving markcheck figure')
        cv2.imwrite(ResultImgPath+'markcheck.bmp', fig)


if __name__ == '__main__':
    FigFolderName = '../../figures/examples/'
    print(readmark('image_203_203_253.bmp'))
    checkmark('image_203_203_253.bmp', OutputMode.FileOutput)
