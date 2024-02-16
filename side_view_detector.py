import dataclasses
import cv2
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class areas:
  """Class to hold area's of each horizontal pressure point by floor.

    Note that in each dictionary, an index "0" equates to the roof, and is in
    order from top down. For example, if there was a 3 story building, index 0
    would be the roof, index 1 would be the upper floor, index 2 would be the
    main floor, and index 3 would be the bottom floor.

    It is possible to not have an index for each floor in every dictionary
    (in fact, this is usually the case).

  Attributes:
    a_area: A dictionary mapping a floors index to the sum of "a" horizontal
      pressure areas.
    b_area: A dictionary mapping a floors index to the sum of "b" horizontal
      pressure areas.
    c_area: A dictionary mapping a floors index to the sum of "c" horizontal
      pressure areas.
    d_area: A dictionary mapping a floors index to the sum of "d" horizontal
      pressure areas.
  """
  a_area: dict[int, float]
  b_area: dict[int, float]
  c_area: dict[int, float]
  d_area: dict[int, float]

  def __str__(self):
    """Converts object to debug string."""
    return f"""
      A areas: {self.a_area}
      B areas: {self.b_area}
      C areas: {self.c_area}
      D areas: {self.d_area}
      """


def detect_interest_region(image: cv2.typing.MatLike, image_len: float):
  width = image.shape[1]
  height = image.shape[0]

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply adaptive thresholding to segment the image
  threshold = cv2.adaptiveThreshold(
      gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

  # Find contours in the thresholded image
  contours, _ = cv2.findContours(
      threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  """
  cv2.imshow('Contours', threshold)
  cv2.waitKey(0)
  """

  # Filter contours based on aspect ratio and area
  min_aspect_ratio = 0.5
  max_aspect_ratio = 30
  min_area = 100
  max_area = width*height * 1
  area_list = []

  text_regions = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = (w * h)
    if aspect_ratio > min_aspect_ratio and area > min_area \
          and aspect_ratio < max_aspect_ratio and area < max_area:
      text_regions.append((x, y, x + w, y + h))
      area_list.append(area)

  # detect the main region
  main_region = text_regions[np.argmax(area_list)]
  main_region = list(main_region)

  # detect right text region
  right_text = []
  right_x = []
  for text_region in text_regions:
    x1, y1, x2, y2 = text_region
    if x1 > 0.75*width and y1 < 0.5 * height:
      right_text.append(text_region)
      right_x.append(x1)

  if len(right_x) > 2:
    main_region[2] = np.min(right_x)

  # detect left text region
  left_text = []
  left_x = []
  for text_region in text_regions:
    x1, y1, x2, y2 = text_region
    if x1 < 0.25*width and y1 < 0.5 * height:
      left_text.append(text_region)
      left_x.append(x1)

  if len(left_x) > 2:
    main_region[0] = np.max(left_x)

  # get the interst region
  image_ = np.ones(image.shape, dtype=np.uint8)*255
  x1, y1, x2, y2 = main_region
  image_[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]

  # Convert the image to grayscale
  gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

  # Apply edge detection to find the edges of the building
  edges = cv2.Canny(gray, 50, 150)

  # Apply line detection
  lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                          threshold=20, minLineLength=20, maxLineGap=10)

  # detect first vertical line and last vertical line
  vertical_x = []
  for line in lines:
    x1, y1, x2, y2 = line[0]
    if np.abs(x1 - x2) < 3:
      vertical_x.append(x1)

  main_region[0] = np.min(vertical_x) - 5
  main_region[1] = main_region[1] - 5
  main_region[2] = np.max(vertical_x) + 5
  main_region[3] = main_region[3] + 5
  # get the interest region

  x1, y1, x2, y2 = main_region
  image_len = image_len*(x2-x1)/image.shape[1]
  image_ = image[y1:y2, x1:x2, :].copy()

  return image_, image_len, [x1, y1]


def detect_left_wall(lines, plate_top, plate_bottom, image, th=0.7, depth=0.15, gap=10, roof=False):
  if lines is None:
      return []

  left_vertical_lines = []
  left_vertical_len = []
  left_vertical_x = []
  left_wall = []

  for line in lines:
    x1, y1, x2, y2 = line[0]
    if np.abs(x1 - x2) < 3 and x1 < image.shape[1]*depth and y1 > plate_top[0][1] - 10 and y1 < plate_bottom[0][1] + 10:
      left_vertical_lines.append(line)

      for line in lines:
        x11, y11, x22, y22 = line[0]
        if y22 > y2 and (y11 - y2) < gap and np.abs(x11 - x22) < 5 and np.abs(x1 - x22) < 5 and y11 > plate_top[0][1]:
          y2 = y22
          left_vertical_lines[-1][0][3] = y2

      left_vertical_x.append(x1)
      left_vertical_len.append(np.abs(y2-y1))

  for i in range(len(left_vertical_x)):
      n = np.argmin(left_vertical_x)
      if left_vertical_len[n] > th * np.abs(plate_top[0][1] - plate_bottom[0][1]):
        left_wall = left_vertical_lines[n]
        break
      left_vertical_x[n] = image.shape[0]
  if len(left_wall) == 0:
    try:
      left_wall = left_vertical_lines[np.argmax(left_vertical_len)]
    except:
      left_wall = []
  if (plate_bottom[0][3] - left_wall[0][3]) < (plate_bottom[0][3] - plate_top[0][3])*0.3 and roof:
    return []
  else:
    return left_wall


def detect_right_wall(lines, plate_top, plate_bottom, image, th=0.7, depth=0.85, gap=10, roof=False):
  if lines is None:
    return []

  right_vertical_lines = []
  right_vertical_len = []
  right_vertical_x = []
  right_wall = []
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      if np.abs(x1 - x2) < 5 and x1 > image.shape[1]*depth and y1 > plate_top[0][1] and y1 < plate_bottom[0][1]:
        right_vertical_lines.append(line)

        for line in lines:
          x11, y11, x22, y22 = line[0]
          if y22 > y2 and (y11 - y2) < gap and np.abs(x11 - x22) < 5 and np.abs(x1 - x22) < 5 and y11 > plate_top[0][1]:
            y2 = y22
            right_vertical_lines[-1][0][3] = y2
        right_vertical_len.append(np.abs(y2-y1))
        right_vertical_x.append(x1)

  for i in range(len(right_vertical_x)):
      n = np.argmax(right_vertical_x)
      if right_vertical_len[n] > th * np.abs(plate_top[0][1] - plate_bottom[0][1]):
        right_wall = right_vertical_lines[n]
        break
      right_vertical_x[n] = 0
  if len(right_wall) == 0:
    right_wall = right_vertical_lines[np.argmax(right_vertical_len)]

  if (plate_bottom[0][3] - right_wall[0][3]) < (plate_bottom[0][3] - plate_top[0][3])*0.3 and roof:
    return []
  else:
    return right_wall


def detect_bottom(lines, image):
  bottom_lines = []
  bottom_len = []
  bottom_y = []
  bottom = []

  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      if np.abs(y1 - y2) < 5 and y1 > image.shape[0]*0.9:
        bottom_lines.append(line)
        bottom_len.append(np.abs(x2-x1))
        bottom_y.append(y1)

  for i in range(len(bottom_y)):
    n = np.argmax(bottom_y)
    if bottom_len[n] > 0.5*image.shape[1]:
      bottom = bottom_lines[n]
      break
    bottom_y[n] = 0
  if len(bottom) == 0:
    bottom = bottom_lines[np.argmax(bottom_len)]
  return bottom


def detect_roof_top(lines, image):
  top_roof_lines = []
  top_roof_len = []

  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      if np.abs(y1 - y2) < 5 and y1 < image.shape[0]*0.2:
        top_roof_lines.append(line)
        top_roof_len.append(np.abs(x2-x1))

  roof_top = top_roof_lines[np.argmax(top_roof_len)]
  return roof_top


def detect_plate_top(lines, roof_top, image):
  plate_top_lines = []
  plate_top_len = []
  plate_top_y = []
  plate_top = []

  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      if np.abs(y1 - y2) < 5 and y1 > image.shape[0]*0.2 and y1 < image.shape[1]*0.4:
        plate_top_lines.append(line)
        plate_top_len.append(np.abs(x2-x1))
        plate_top_y.append(y1)

  for i in range(len(plate_top_y)):
    n = np.argmin(plate_top_y)
    if plate_top_len[n] > 0.15*image.shape[1] and np.abs(plate_top_y[n] - roof_top[0][1]) > 0.05*image.shape[0] and\
        ((np.abs(plate_top_lines[n][0][0] - roof_top[0][0]) < 30 or np.abs(plate_top_lines[n][0][2] - roof_top[0][2])) < 30):
      plate_top = plate_top_lines[n]
      break
    plate_top_y[n] = image.shape[0]
  if len(plate_top) == 0:
    plate_top = plate_top_lines[np.argmax(plate_top_len)]
  return plate_top


# TODO: Add doc comment and variable types in parameters. Also add return type.
def detect_roof_rest(lines, roof_top, roof_left, roof_right, plate_top, image, roof_rest_, nn):
  """_summary_

  Args:
      lines (_type_): _description_
      roof_top (_type_): _description_
      roof_left (_type_): _description_
      roof_right (_type_): _description_
      plate_top (_type_): _description_
      image (_type_): _description_
      roof_rest_ (_type_): _description_
      nn (_type_): _description_

  Returns:
      _type_: _description_
  """
  if roof_top[0][2] - roof_top[0][0] > 0.97*(roof_right[0][0] - roof_left[0][0]):
    return roof_rest_

  roof_rest_lines = []
  roof_rest_len = []
  roof_rest_y = []
  roof_rest = roof_rest_.copy()

  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]

      if abs(roof_top[0][0] - roof_left[0][0]) > 20:
        if np.abs(y1 - y2) < 5 and y1 > roof_top[0][1] and y1 < plate_top[0][1] \
              and x1 < roof_top[0][0] - 10 and x2 < roof_top[0][0] + 10:
          roof_rest_lines.append(line)
          roof_rest_len.append(np.abs(x2-x1))
          roof_rest_y.append(y1)

      if abs(roof_top[0][2] - roof_right[0][2]) > 20:
          if np.abs(y1 - y2) < 5 and y1 > roof_top[0][1] and y1 < plate_top[0][1] \
              and x2 > roof_top[0][2] + 10 and x1 > roof_top[0][2] - 10:
            roof_rest_lines.append(line)
            roof_rest_len.append(np.abs(x2-x1))
            roof_rest_y.append(y1)

  for i in range(len(roof_rest_y)):
      n = np.argmin(roof_rest_y)
      if abs(roof_top[0][0] - roof_left[0][0]) > 20:
          if roof_rest_len[n] > 0.25*(abs(roof_top[0][0] - roof_left[0][0])):
              roof_rest.append(roof_rest_lines[n])
              break
      if abs(roof_top[0][2] - roof_right[0][2]) > 20:
          if roof_rest_len[n] > 0.25*(abs(roof_top[0][2] - roof_right[0][2])):
              roof_rest.append(roof_rest_lines[n])
              break
      roof_rest_y[n] = image.shape[0]
  if len(roof_rest) == len(roof_rest_):
      roof_rest = roof_rest_lines[np.argmax(roof_rest_len)]

  if abs(roof_top[0][2] - roof_right[0][2]) > 20:
      roof_top[0][2] = roof_rest[-1][0][2]
  if abs(roof_top[0][0] - roof_left[0][0]) > 20:
      roof_top[0][0] = roof_rest[-1][0][0]
  nn = nn + 1
  if nn > 5:
      return roof_rest
  else:
      roof_rest = detect_roof_rest(
          lines, roof_top, roof_left, roof_right, plate_top, image, roof_rest, nn)
  return roof_rest


# TODO: Add doc comment and variable types in parameters. Also add return type.
def get_area(roof_points, x1, x2, y):
  """_summary_

  Args:
      roof_points (_type_): _description_
      x1 (_type_): _description_
      x2 (_type_): _description_
      y (_type_): _description_

  Returns:
      _type_: _description_
  """
  n1 = np.where(roof_points[:, 0] <= x1)[0][-1]
  y1 = roof_points[n1, 1] + (roof_points[n1+1, 1] - roof_points[n1, 1])*(x1 - roof_points[n1, 0])\
      / (roof_points[n1+1, 0] - roof_points[n1, 0])

  n2 = np.where(roof_points[:, 0] <= x2)[0][-1]
  try:
    y2 = (roof_points[n2, 1] + (roof_points[n2+1, 1] -
                                roof_points[n2, 1])*(x2 - roof_points[n2, 0]) /
          (roof_points[n2+1, 0] - roof_points[n2, 0]))
  except:
    y2 = roof_points[n2, 1]

  points = []
  points.append([x1, y1])
  for k in range(n1+1, n2-1):
    points.append([roof_points[k][0], roof_points[k][1]])

  points.append([x2, y2])
  points = np.array(points)

  area = 0

  for k in range(len(points)-1):
    area = area + (points[k+1][0] - points[k][0]) * \
      (y - (points[k+1][1] + points[k][1])/2)
  return area


# TODO: Add comments whenever there is some sort of multiplier describing what
# it is and why.
def get_horizontal_pressure_areas_on_x(image_path: str, output_path: str,  image_length: float
                                       ) -> areas:
  """Gets the sum of all the pressure areas for each floor.

  Args:
    image_path: A path to the image being detected on a local machine
      (full path). The image is always in the form of .png.
    image_length: The length of the image in real life scale in feet (from x==0
      to x==len(x-1)).
    output_path : A path to be saved.

  Returns:
    An object holding dictionaries mapping each floors index to their
      respective a, b, c, and d areas.
  """
  # Load the image
  image1 = cv2.imread(image_path)

  # detect interested region
  image2, image_len, p0 = detect_interest_region(image1.copy(), image_length)

  width = image2.shape[1]
  height = image2.shape[0]

  ratio = 1050/width
  height2 = int(height*ratio)

  image = cv2.resize(image2, (1050, height2))

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray[np.where(gray < 150)] = 0

  # Apply edge detection to find the edges of the building
  edges = cv2.Canny(gray, 50, 150)

  """
  cv2.imshow('edges', edges)
  cv2.waitKey(0)
  """

  # Apply line detection
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20,
                          maxLineGap=10)

  for n, line in enumerate(lines):
    x1, y1, x1, y2 = line[0]
    if y1 > y2:
      line[0][1] = y2
      line[0][3] = y1
      lines[n] = line

  # ==== detect the top of roof ===
  roof_top = detect_roof_top(lines, image)
  x1, y1, x2, y2 = roof_top[0]
  cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

  # ==== detect the bottom ===
  bottom = detect_bottom(lines, image)

  x1, y1, x2, y2 = bottom[0]
  cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 1)

  # ==== detect the top of plate ===
  plate_top = detect_plate_top(lines, roof_top.copy(), image)

  x1, y1, x2, y2 = plate_top[0]
  cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1)

  # =====  detect roof left and roof right =====
  roof_left = detect_left_wall(
    lines, roof_top.copy(), plate_top.copy(), image, th=0.25)

  x1, y1, x2, y2 = roof_left[0]
  cv2.line(image, (x1, y1), (x2, y2), (200, 200, 0), 1)

  # right
  roof_right = detect_right_wall(
    lines, roof_top.copy(), plate_top.copy(), image, th=0.25)

  x1, y1, x2, y2 = roof_right[0]
  cv2.line(image, (x1, y1), (x2, y2), (200, 200, 0), 1)

  # === detect the rest of roof ===
  roof_rest = detect_roof_rest(lines, roof_top.copy(), roof_left.copy(),
                               roof_right.copy(), plate_top.copy(), image,
                               [], 0)

  for k in range(len(roof_rest)):
    x1, y1, x2, y2 = roof_rest[k][0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

  # ====== detect left wall =====
  left_wall = detect_left_wall(
    lines, plate_top.copy(), bottom.copy(), image, gap=50)

  x1, y1, x2, y2 = left_wall[0]
  cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

  # ====== detect right  wall =====
  right_wall = detect_right_wall(lines, plate_top.copy(), bottom.copy(),
                                 image, gap=50)

  x1, y1, x2, y2 = right_wall[0]
  cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

  # ====== detect left space and right space
  left_roof = []
  right_roof = []

  if left_wall[0][0] > 0.08*image.shape[1]:
      try:
          left_roof = detect_left_wall(lines, plate_top.copy(), bottom.copy(),
                                       image, th=0.15, depth=0.05, roof=True)

          x1, y1, x2, y2 = left_roof[0]
          cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
      except:
        pass

  if right_wall[0][0] < 0.92*image.shape[1]:
    try:
      right_roof = detect_right_wall(lines, plate_top.copy(), bottom.copy(),
                                     image, th=0.15, depth=0.95, roof=True)
      x1, y1, x2, y2 = right_roof[0]
      cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    except:
      pass

  # ===========
  """
  # Display the image with the detected lines
  cv2.imshow("Roof Plates", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  """

  """
  # Draw the detected lines on the image
  if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          
          cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
      # Display the image with the detected lines
  
  cv2.imshow("lines", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  """

  roof_list = roof_rest.copy()
  roof_list.append(roof_top)

  # calculate the total square footage of the area labeled for each letter
  # get roof points

  roof_points = []
  roof_points.append([roof_left[0][0], roof_left[0][1]])

  while len(roof_list) > 0:
    dist = []
    for k in range(len(roof_list)):
      d = np.sqrt((roof_points[-1][0]-roof_list[k][0][0])
                  ** 2 + (roof_points[-1][1] - roof_list[k][0][0])**2)
      dist.append(d)
    n = np.argmin(dist)
    roof_points.append([roof_list[n][0][0], roof_list[n][0][1]])
    roof_points.append([roof_list[n][0][2], roof_list[n][0][3]])
    roof_list.pop(n)

  roof_points = np.array(roof_points)

  left_bottom = [left_wall[0][0], bottom[0][1]]
  right_bottom = [right_wall[0][0], bottom[0][1]]
  plate_top_left = [left_wall[0][0], plate_top[0][1]]
  plate_top_right = [right_wall[0][0], plate_top[0][1]]  # TODO: Not used?

  p1 = [left_bottom[0] + 0.2 *
        (right_bottom[0] - left_bottom[0]), left_bottom[1]]
  p2 = [left_bottom[0] + 0.8 *
        (right_bottom[0] - left_bottom[0]), bottom[0][1]]

  if (bottom[0][2] - bottom[0][0]) / (right_bottom[0] - left_bottom[0]) < 0.6:
    A1 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.165*(left_bottom[1] - plate_top_left[1])
    A2 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.165*(left_bottom[1] - plate_top_left[1])

    A3 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.33*(left_bottom[1] - plate_top_left[1])
    A4 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.33*(left_bottom[1] - plate_top_left[1])

    C1 = 0.6 * (right_bottom[0] - left_bottom[0]) * \
      0.165*(left_bottom[1] - plate_top_left[1])
    C2 = 0.6 * (right_bottom[0] - left_bottom[0]) * \
      0.33*(left_bottom[1] - plate_top_left[1])

  else:
    A1 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.25*(left_bottom[1] - plate_top_left[1])
    A2 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.25*(left_bottom[1] - plate_top_left[1])

    A3 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.5*(left_bottom[1] - plate_top_left[1])
    A4 = 0.2 * (right_bottom[0] - left_bottom[0]) * \
      0.5*(left_bottom[1] - plate_top_left[1])

    C1 = 0.6 * (right_bottom[0] - left_bottom[0]) * \
      0.25*(left_bottom[1] - plate_top_left[1])
    C2 = 0.6 * (right_bottom[0] - left_bottom[0]) * \
      0.5*(left_bottom[1] - plate_top_left[1])

  D = (plate_top[0][1] - roof_top[0][1]) * (p2[0] - p1[0])

  # area of B1
  x1 = roof_left[0][0]
  y = plate_top[0][1]
  x2 = p1[0]
  B1 = get_area(roof_points.copy(), x1, x2, y)

  # area of B2
  x1 = p2[0]
  x2 = roof_right[0][0]
  B2 = get_area(roof_points.copy(), x1, x2, y)

  # area of D
  x1 = p1[0]
  x2 = p2[0]
  D = get_area(roof_points.copy(), x1, x2, y)

  B3 = 0
  if len(left_roof) > 0:
      B3 = B3 + (left_roof[0][3] - left_roof[0][1]) * \
          (left_bottom[0] - left_roof[0][0])

  if len(right_roof) > 0:
    B3 = B3 + (right_roof[0][3] - right_roof[0][1]) * \
      (right_roof[0][0] - right_bottom[0])

  conv = (image_len/image.shape[1])**2
  A = (A1 + A2 + A3 + A4)*conv
  B = (B1 + B2 + B3)*conv
  C = (C1 + C2)*conv

  # TODO(ryan): Flipped this so that roof is 1 and upper is 2.
  # Let me know if that is incorrect.
  a_area = {1: (A1 + A2)*conv, 2: (A3 + A4)*conv}
  b_area = {1: (B1 + B2)*conv, 2: B3*conv}
  c_area = {1: C1*conv, 2: C2*conv}
  d_area = {1: D*conv}

  # drawing
  vert1 = [p1, [p1[0], 0]]
  vert2 = [p2, [p2[0], 0]]
  if (bottom[0][2] - bottom[0][0]) / (right_bottom[0] - left_bottom[0]) < 0.6:
    horizon1 = [[0, plate_top[0][1] + 0.165*(bottom[0][1] - plate_top[0][1])], [
      image.shape[1], plate_top[0][1] + 0.165*(bottom[0][1] - plate_top[0][1])]]
    horizon2 = [[0, plate_top[0][1] + 0.5*(bottom[0][1] - plate_top[0][1])], [
      image.shape[1], plate_top[0][1] + 0.5*(bottom[0][1] - plate_top[0][1])]]
  else:
    horizon1 = [[0, plate_top[0][1] + 0.25*(bottom[0][1] - plate_top[0][1])], [
      image.shape[1], plate_top[0][1] + 0.25*(bottom[0][1] - plate_top[0][1])]]
    horizon2 = [[0, plate_top[0][1] + 0.75*(bottom[0][1] - plate_top[0][1])], [
      image.shape[1], plate_top[0][1] + 0.75*(bottom[0][1] - plate_top[0][1])]]

  vert1 = np.array(vert1)
  vert2 = np.array(vert2)

  horizon1 = np.array(horizon1)
  horizon2 = np.array(horizon2)

  r_lines = []

  vert1 = (vert1/ratio).astype(int)
  vert1[:, 0] = vert1[:, 0] + p0[0]
  vert1[:, 1] = vert1[:, 1] + p0[1]
  r_lines.append([vert1[0, 0], vert1[0, 1], vert1[1, 0], vert1[1, 1]])

  vert2 = (vert2/ratio).astype(int)
  vert2[:, 0] = vert2[:, 0] + p0[0]
  vert2[:, 1] = vert2[:, 1] + p0[1]
  r_lines.append([vert2[0, 0], vert2[0, 1], vert2[1, 0], vert2[1, 1]])

  horizon1 = (horizon1/ratio).astype(int)
  horizon1[:, 0] = horizon1[:, 0] + p0[0]
  horizon1[:, 1] = horizon1[:, 1] + p0[1]
  r_lines.append([horizon1[0, 0], horizon1[0, 1],
                  horizon1[1, 0], horizon1[1, 1]])

  horizon2 = (horizon2/ratio).astype(int)
  horizon2[:, 0] = horizon2[:, 0] + p0[0]
  horizon2[:, 1] = horizon2[:, 1] + p0[1]
  r_lines.append([horizon2[0, 0], horizon2[0, 1],
                  horizon2[1, 0], horizon2[1, 1]])

  if lines is not None:
    for line in r_lines:
      x1, y1, x2, y2 = line
      cv2.line(image1, (x1, y1), (x2, y2), (0, 0, 255), 2)

  image1 = cv2.circle(
    image1, (vert1[0, 0], vert1[0, 1]), radius=5, color=(0, 0, 255), thickness=-1)
  image1 = cv2.circle(
    image1, (vert2[0, 0], vert2[0, 1]), radius=5, color=(0, 0, 255), thickness=-1)

  """
  cv2.imshow("lines", image1)
  cv2.waitKey(0)
  cv2.destroyAllWindows() 
  """
  # save image file
  filename = output_path + "/" + image_path.split("/")[-1]
  cv2.imwrite(filename, image1)

  return areas(
    a_area=a_area,
    b_area=b_area,
    c_area=c_area,
    d_area=d_area
  )

image_path = "Side Elevations/left_1.png"
# image_path = "Side Elevations/left_2.png"
# image_path = "Side Elevations/right_1.png"
# image_path = "Side Elevations/right_2.png"
# image_path = "Side Elevations/right_3.png"


output_path = 'side_view_detect_output'

image_length = 44.2
all_areas = get_horizontal_pressure_areas_on_x(
    image_path, output_path, image_length)

print(all_areas)
