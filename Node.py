class Node:
    """
    Khởi tạo một node của cây quyết định
    feature : feature được của node
    thresdhold : ngưỡng để rẽ nhánh cho node
    data_left : dữ liệu nhánh dưới bên trái
    data_right : dữ liệu nhánh dưới bên phải
    gini : Chỉ số gini của node
    value : Giá trị của kết quả (nếu như đây là node lá thì value là giá trị cần tìm)
    """
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gini=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gini = gini
        self.value = value