from pathlib import Path
from typing import Optional

def first_match(folder: Path, pattern: str) -> Optional[Path]:
    """查找文件夹中第一个匹配的文件"""
    hits = list(folder.glob(pattern))
    return hits[0] if hits else None

def locate_stim_and_dem(folder: Path) -> tuple[Path, Optional[Path]]:
    """
    定位.stim和.dem文件
    :param folder: 包含电路文件的目录
    :return: (stim_path, dem_path) 元组
    :raises FileNotFoundError: 如果找不到.stim文件
    """
    stim_path = (folder / "circuit_noisy.stim") if (folder / "circuit_noisy.stim").exists() \
        else first_match(folder, "*.stim")
    
    if stim_path is None:
        raise FileNotFoundError(f"No .stim files found in {folder}")
    
    dem_path = first_match(folder, "*.dem")
    return stim_path, dem_path

def test_file_finder():
    """测试文件查找功能"""
    test_dir = Path("C:\\Users\\Lenovo\\Downloads\\google_qec3v5_experiment_data\\surface_code_bZ_d3_r25_center_5_3")
    try:
        stim, dem = locate_stim_and_dem(test_dir)
        print(f"Found stim: {stim}")
        print(f"Found dem: {dem if dem else 'None'}")
    except FileNotFoundError as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_file_finder()