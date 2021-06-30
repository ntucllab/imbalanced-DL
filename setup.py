from setuptools import setup


setup(
    name='imbalanceddl',
    version='0.0.1',
    description='Deep Imbalanced Learning in Python',
    packages=[
        'imbalanceddl',
        'imbalanceddl.strategy',
        'imbalanceddl.net',
        'imbalanceddl.loss',
        'imbalanceddl.dataset',
        'imbalanceddl.utils',
    ],
    package_dir={
        'imbalanceddl': 'imbalanceddl',
        'imbalanceddl.strategy': 'imbalanceddl/strategy',
        'imbalanceddl.net': 'imbalanceddl/net',
        'imbalanceddl.loss': 'imbalanceddl/loss',
        'imbalanceddl.dataset': 'imbalanceddl/dataset',
        'imbalanceddl.utils': 'imbalanceddl/utils',
    },
)
