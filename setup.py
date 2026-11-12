from setuptools import setup, find_packages

setup(
    name="my_nnunet",  # 包名称
    version="0.1.0",  # 版本号
    description="Segmentation network",  # 简要描述
    author="shi hang",  # 作者信息
    author_email="shihang@ahut.edu.cn",  # 作者邮箱
    packages=find_packages(),  # 自动查找所有包含 `__init__.py` 的模块
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python版本要求
)
