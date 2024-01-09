import numpy as np


def eval_metric_neck(img, neck_center):
    """
    Evaluate the metric of the neck position
    :param img: the image
    :param neck_center: (tuple) the neck center
    """
    line_y = int(img.shape[0]/3)
    score = abs(line_y - neck_center[1]) / line_y

    return score


def eval_metric_bottom(img, pelvis, knee=None, ankle=None):
    """
    Evaluate the metric of the bottom position
    :param img: the image
    :param pelvis: (tuple) the pelvis
    :param knee: (tuple) the knee
    :param ankle: (tuple) the ankle
    """
    line_y = int(img.shape[0]/3)
    
    if ankle == None: # 발목 좌표가 없는 경우 (무릎까지 나온 경우)
        y_knee = img.shape[0] - knee[1]
        knee_pelvis = knee[1] - pelvis[1]
        score = abs(int(knee_pelvis/2) - y_knee) / line_y
    
    else: # 발목까지 모두 나온 경우
        score = abs(img.shape[0] - ankle[1]) / line_y
        
    return score


def eval_metric_eye(image, left_eye, right_eye):
    """
    Evaluate the metric of the eye position
    :param image: the image
    :param left_eye: (tuple) the left eye
    :param right_eye: (tuple) the right eye
    """
    height, width, _ = image.shape

    left_eye_x = left_eye[0]
    right_eye_x = right_eye[0]
    left_eye_y = left_eye[1]
    right_eye_y = right_eye[1]

    line_y = int(image.shape[0] / 3)

    # 삼분할선 계산
    second_height = 2 * height // 3  # 상단 가로선

    first_width = width // 3  # 왼쪽 세로선
    second_width = 2 * width // 3  # 오른쪽 세로선

    # 눈의 y 좌표
    eye_y = (left_eye_y + right_eye_y) / 2

    abnormal_score = 0
    
    # 눈 위치 가운데 블록에 위치하도록
    if left_eye_x < first_width:
        abnormal_score += abs(first_width-left_eye_x) / line_y
    if right_eye_x > second_width:
        abnormal_score += abs(second_width-right_eye_x) / line_y

    abnormal_score += abs(second_height-eye_y) / line_y
    
    return abnormal_score


def find_neck_center(shoulder_points, nose_point):
    """
    Find the neck center
    :param shoulder_points: (tuple) the shoulder points
    :param nose_point: (tuple) the nose point
    """
    shoulder_center = ((shoulder_points[0][0] + shoulder_points[1][0]) / 2, 
                       (shoulder_points[0][1] + shoulder_points[1][1]) / 2)

    neck_center = ((shoulder_center[0] + nose_point[0]) / 2, 
                   (shoulder_center[1] + nose_point[1]) / 2)
    
    return neck_center


def calculate_new_coordinates(knee=None, 
                              ankle=None, 
                              pelvic=(0, 0),
                              neck_center=(0, 0)):
    """
    Calculate the new coordinates based on the neck center
    :param neck_center: (tuple) the base neck center coordinates
    :param knee: (tuple) the knee coordinates
    :param ankle: (tuple) the ankle coordinates
    :param pelvic: (tuple) the pelvic coordinates
    """
    
    neck_center_x, neck_center_y = neck_center
    knee_x, knee_y = knee
    ankle_x, ankle_y = ankle
    pelvic_x, pelvic_y = pelvic

    # Calculate the distance between the original neck center and the base neck center
    dx = neck_center_x - (knee_x + ankle_x + pelvic_x) / 3
    dy = neck_center_y - (knee_y + ankle_y + pelvic_y) / 3

    # Calculate the new coordinates by applying the parallel translation
    new_knee_x = knee_x + dx
    new_knee_y = knee_y + dy
    new_pelvic_x = pelvic_x + dx
    new_pelvic_y = pelvic_y + dy

    if ankle is not None:
        new_ankle_x = ankle_x + dx
        new_ankle_y = ankle_y + dy

        return (new_knee_x, new_knee_y), (new_ankle_x, new_ankle_y), (new_pelvic_x, new_pelvic_y)
    else:
        return (new_knee_x, new_knee_y), None, (new_pelvic_x, new_pelvic_y)


def find_best_neck_pos(shoulder_points: tuple = (0, 0), 
                       mouth_point: tuple = (0, 0), 
                       pelvic: tuple = (0, 0), 
                       knee: tuple = None, 
                       ankle: tuple = None, 
                       img = None):
    """
    Find the best neck position
    :param shoulder_points: (tuple) the shoulder points
    :param mouth_point: (tuple) the mouth point
    :param img: the image
    """
    img_width = img.shape[1]
    first_thrid_height = int(img.shape[0]/3)

    best_score_neck = float('inf')
    best_score_bottom = float('inf')
    neck_center_x, neck_center_y = find_neck_center(shoulder_points, mouth_point)

    for y in np.arange(first_thrid_height - 0.5, first_thrid_height + 0.5, 0.1):
        for x in range(img_width):
            # knee와 ankle과 pelvic의 경우 왼왼쪽 다리에 해당하는 좌표만 입력
            knee_pos, ankle_pos, pelvic_pos = calculate_new_coordinates(knee, ankle, pelvic, (neck_center_x, neck_center_y))
            neck_score = eval_metric_neck(img, (x, y))
            bottom_score = eval_metric_bottom(img, pelvic_pos, knee_pos, ankle_pos)

            if neck_score < best_score_neck & bottom_score < best_score_bottom:
                best_score_neck = neck_score
                best_score_bottom = bottom_score
                optimal_center_neck = (x, y)

    return optimal_center_neck, best_score_neck, best_score_bottom


def find_best_eye_pos(left_eye_pos, right_eye_pos, img):
    """
    Find the best eye position
    :param left_eye_pos: (tuple) the left eye position
    :param right_eye_pos: (tuple) the right eye position
    :param img: the image
    """
    img_width = img.shape[1]
    first_thrid_height = int(img.shape[0]/3)

    left_eye_pos_x, _ = left_eye_pos
    right_eye_pos_x, _ = right_eye_pos

    eye_distance = right_eye_pos_x - left_eye_pos_x

    best_score = float('inf')
    optimal_left_eye = left_eye_pos
    optimal_right_eye = right_eye_pos

    for y_offset in np.arange(first_thrid_height - 0.5, first_thrid_height + 0.5, 0.1):
        for x_offset in range(img_width):
            new_left_eye_x = left_eye_pos_x + x_offset
            new_right_eye_x = new_left_eye_x + eye_distance

            if 0 <= new_left_eye_x < img_width and 0 <= new_right_eye_x < img_width:
                score = eval_metric_eye(img, (new_left_eye_x, y_offset), (new_right_eye_x, y_offset))

                if score < best_score:
                    best_score = score
                    optimal_left_eye = (new_left_eye_x, y_offset)
                    optimal_right_eye = (new_right_eye_x, y_offset)

    return optimal_left_eye, optimal_right_eye, best_score


def main_process(img, 
                 shoulder_points, 
                 mouth_point, 
                 left_eye_pos, 
                 right_eye_pos, 
                 pelvic,
                 knee: tuple = None, 
                 ankle: tuple = None, 
                 knee_label: int = None):
    """
    Main process
    :param shoulder_points: (tuple) the shoulder points
    :param mouth_point: (tuple) the mouth point
    :param left_eye_pos: (tuple) the left eye position
    :param right_eye_pos: (tuple) the right eye position
    :param pelvic: (tuple) the pelvic position
    :param knee: (tuple) the knee position
    :param ankle: (tuple) the ankle position
    :param img: the image
    """
    # 왼쪽 다리에 대해 무릎의 레이블이 있는 경우 전신으로 계산
    if knee_label is not None and knee is not None:
        first_score_neck = eval_metric_neck(img, find_neck_center(shoulder_points, mouth_point))
        first_score_bottom = eval_metric_bottom(img, pelvic, knee, ankle)

        optimal_center_neck, best_score_neck, best_score_bottom = find_best_neck_pos(shoulder_points, mouth_point, pelvic, knee, ankle, img)

        return optimal_center_neck, best_score_neck, best_score_bottom, first_score_neck, first_score_bottom
    else:
        first_score_eye = eval_metric_eye(img, left_eye_pos, right_eye_pos)
        optimal_left_eye, optimal_right_eye, best_score_eye = find_best_eye_pos(left_eye_pos, right_eye_pos, img)

        return optimal_left_eye, optimal_right_eye, best_score_eye, first_score_eye
