import cv2 as cv
TxTFileName = 'E://batch mark result.txt'
FigFolderName = 'E://imgimp//imim//'



def ReadMark(figname):
    """
    :param figname: 图片名称 典型值'image_178_3_153.bmp'
    :return: 数组list 典型值[[-600, 1800, 1141.3, 298.25], [-600, 1500, 1085.8, 430.25]]
    数组的长度表示标注点的数量 每个标注点包含四个参数 X Y x y 大写为真实坐标 小写为图像坐标
    xy为小数是因为标注时图片太大经过缩放 产生了小数坐标标注 参考小数坐标寻找最临近亮点即可
    """
    head,alpha,beta,gamaend = figname.split('_')
    gama,end = gamaend.split('.')
    # 转整数
    alpha = int(alpha)
    beta = int(beta)
    gama = int(gama)

    with open(TxTFileName, 'r', encoding='UTF-8') as f:
        result = []
        lenth = 0
        for i, line in enumerate(f):
            X, Y, x, y,alphahere,betahere,gamahere,num = line.split('\t')
            X = int(X)
            Y = int(Y)
            x= float(x)
            y= float(y)
            alphahere = int(alphahere)
            betahere = int(betahere)
            gamahere = int(gamahere)
            num = int(num)
            if alphahere==alpha and betahere==beta and gamahere==gama:
                result.append([X,Y,x,y])
                lenth = num
        if not len(result)==lenth:
            print('error NOOOOOOO!')
            return None
        else:
            return result

def change(x,y,xsize,ysize):
    if x <= 0:
        x = 1
    if y <= 0:
        y = 1
    if x >= xsize:
        x = xsize
    if y >= ysize:
        y = ysize
    return x,y

def SeeSee(figname):
    """
    输入图片名称 显示检查标注点 会打开一张图片
    :param figname:图片名称
    :return:无
    """
    figname = FigFolderName+figname
    fig = cv.imread(figname)
    xsize = fig.shape[1]
    ysize = fig.shape[0]
    # 拿到标注点信息
    points = ReadMark(figname)
    font = cv.FONT_HERSHEY_SIMPLEX
    # 标点 画圈 写字
    if len(points):
        for point in points:
            p = change(int(point[2]), int(point[3]),xsize,ysize)
            cv.circle(fig, (p[0],p[1]), 15, (0, 0, 255), 2)
            p = change(int(point[2]), int(point[3])-10,xsize,ysize)
            cv.putText(fig, str(int(point[0]))+','+str(int(point[1])),
                       (p[0],p[1]), font, 1,(255, 255, 255), 1)
    cv.namedWindow('see_image', cv.WINDOW_AUTOSIZE)
    # 缩放 防止超出屏幕
    size = fig.shape
    fig = cv.resize(fig, (int(size[1] * 0.5), int(size[0] * 0.5)), cv.INTER_LINEAR)
    # 显示
    cv.imshow('see_image', fig)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    print(ReadMark('image_153_3_103.bmp'))
    SeeSee('image_153_3_103.bmp')