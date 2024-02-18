from setuptools import find_packages,setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list
    '''
    with open(file_path) as f:
        requirements = f.read().splitlines()
    

    return requirements

 
    


setup(

    name='MLPROJECT-GENERIC',
    version='0.0.1',
    author='Hassan Taj',
    author_email='hassantaj00@gmail.com',
    packages= find_packages(),
    install_requires=get_requirements('requirements.txt')

)