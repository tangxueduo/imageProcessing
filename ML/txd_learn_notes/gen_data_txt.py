import os


def gen_txt_file(root_dir, txt_path):
    """制作训练数据集的txt,标签文件
    Args:
        root_dir: 训练集/测试集/验证集 目录
        txt_path: 生成的txt文件路径
    """
    if not os.path.isfile(txt_path):
        return
    for file in os.listdir(root_dir):
        label = 0
        img_file = os.path.join(root_dir, file)
        print(img_file)
        if "dog" in img_file:
            label = 1
        with open(txt_path, "a") as fp:
            fp.write(f"{img_file} {label}\n")


def main():
    gen_txt_file(
        root_dir="/media/tx-deepocean/Data/DICOMS/demos/torch_datasets/cats",
        txt_path="/media/tx-deepocean/Data/DICOMS/demos/Projects/pytorch-tutorial/txd_learn_notes/test.txt",
    )


if __name__ == "__main__":
    main()
