import seaborn as sns
import matplotlib.pyplot as plt


def ShowInfo(data):
    print((data.head()), '\n')
    print(data.isnull().sum())

    '''
        --Describe()--
    - Xóa giá trị Null để tránh lỗi
    - List phần trăm
    - List dtypes cần xét
    - Describe data
    '''
    data.dropna(inplace=True)
    perc = [.20, .40, .60, .80]
    include = ['object', 'float', 'int']
    describe = data.describe(percentiles=perc, include=include)
    print(describe, '\n')

    # Số liệu thống kê tóm tắt
    data.info()


def ShowFig(new_data):
    # Đánh giá tương quan và show heatmap(biểu đồ tác động)
    corr = new_data.corr()
    plt.figure(figsize=(10, 8))
    # Heatmap() để vẽ biểu đồ tương quan, annot=True: để in quá trị vào trong bảng
    sns.heatmap(corr, annot=True)

    # Phan phoi du doan va bien response
    plt.figure(figsize=(10, 8))
    sns.distplot(new_data['charges'])
    plt.title('Charges Distribute')

    plt.figure(figsize=(10, 8))
    sns.distplot(new_data['bmi'])
    plt.title('BMT Distribute')

    # Bieu do LR
    # plt.figure(figsize=(10, 8))
    # sns.regplot(x='charges', y='bmi', data=data.head(500))
    # plt.title('Linear Relationship')
    plt.show()
