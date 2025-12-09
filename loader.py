import numpy as np
import os
import glob

def load_bin_sequence(folder_path):
    """
    递归读取文件夹及其子文件夹中的 .bin 点云序列。
    """
    frames = []
    
    if not os.path.exists(folder_path):
        return []

    # [修改] 使用 recursive=True 递归查找所有 .bin 文件
    # 例如：Folder/SubFolder/frame.bin 也能被找到
    bin_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.bin"), recursive=True))
    
    # 如果没找到，打印一下以便调试
    if not bin_files:
        # print(f"  [Loader] 在 {os.path.basename(folder_path)} 中未找到 .bin 文件")
        return []

    for bin_file in bin_files:
        try:
            # 读取 float32
            raw_data = np.fromfile(bin_file, dtype=np.float32)
            
            # 形状重塑
            if raw_data.size % 4 == 0:
                points = raw_data.reshape(-1, 4)[:, :3] 
            elif raw_data.size % 3 == 0:
                points = raw_data.reshape(-1, 3)
            else:
                continue 
                
            frames.append(points)
        except Exception:
            continue
            
    return frames