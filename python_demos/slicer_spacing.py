import numpy as np

spacing = np.array([0.558594, 0.558594, 0.625])
origin = np.array([0, 0, 0])
arr = np.eye((4))

IJKToRASDirections = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
#
def GetIJKToRASMatrix(arr):
    for row in range(3):
        for col in range(3):
            arr[row][col] = spacing[col] * IJKToRASDirections[row][col]
        arr[row][3] = origin[row]
    return arr


print(GetIJKToRASMatrix(arr))


#  Compute slice spacing from the volume axis closest matching the slice axis, projected to the slice axis.

#   必须是线性变换
#   vtkMRMLTransformNode *transformNode = volumeNode->GetParentTransformNode();


def get_spacing():
    volumeRASToWorld = transformNode.GetMatrixTransformToWorld(
        volumeRASToWorld
    )  # 333333
    ijkToWorld = np.multiply(volumeRASToWorld, ijkToWorld, ijkToWorld)
    SliceSpacing = np.zeros(1, 3)
    worldToIJK = np.invert(ijkToWorld)
    sliceToRAS = [
        -0.923919,
        0,
        -0.382588,
        0.257984,
        0,
        1,
        0,
        -6.81277,
        -0.382588,
        0,
        0.923919,
        60.5079,
        0,
        0,
        0,
        1,
    ]
    sliceToIJK = np.multiply(worldToIJK, sliceToRAS, sliceToIJK)  # 111111
    ijkToSlice = np.invert(sliceToIJK)

    # scale = [] # 3维的
    # 将矩阵左上角的3*3方阵 列归一化
    # vtkAddonMathUtilities::NormalizeOrientationMatrixColumns(sliceToIJK, scale); # 22222
    #   // after normalization, sliceToIJK only contains slice axis directions

    for sliceAxisIndex in range(3):
        # // Slice axis direction in IJK coordinate system
        sliceAxisDirection_I = abs(sliceToIJK.GetElement(0, sliceAxisIndex))
        sliceAxisDirection_J = abs(sliceToIJK.GetElement(1, sliceAxisIndex))
        sliceAxisDirection_K = abs(sliceToIJK.GetElement(2, sliceAxisIndex))
    if sliceAxisDirection_I > sliceAxisDirection_J:
        if sliceAxisDirection_I > sliceAxisDirection_K:
            # // this sliceAxis direction is closest volume I axis direction
            SliceSpacing[sliceAxisIndex] = abs(
                ijkToSlice.GetElement(sliceAxisIndex, 0)
            )  # I));
        else:
            # // this sliceAxis direction is closest volume K axis direction
            SliceSpacing[sliceAxisIndex] = abs(
                ijkToSlice.GetElement(sliceAxisIndex, 2)
            )  # K));
    else:
        if sliceAxisDirection_J > sliceAxisDirection_K:
            # // this sliceAxis direction is closest volume J axis direction
            SliceSpacing[sliceAxisIndex] = abs(
                ijkToSlice.GetElement(sliceAxisIndex, 1)
            )  # J));
        else:
            # // this sliceAxis direction is closest volume K axis direction
            SliceSpacing[sliceAxisIndex] = abs(
                ijkToSlice.GetElement(sliceAxisIndex, 2)
            )  # K));
    return SliceSpacing
