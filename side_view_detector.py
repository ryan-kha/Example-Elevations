import dataclasses

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


def get_horizontal_pressure_areas_on_x(image_path: str, image_length: float
                                       ) -> areas:
  """Gets the sum of all the pressure areas for each floor.

  Args:
    image_path: A path to the image being detected on a local machine
      (full path). The image is always in the form of .png.
    image_length: The length of the image in real life scale in feet (from x==0
      to x==len(x-1)).

  Returns:
    An object holding dictionaries mapping each floors index to their
      respective a, b, c, and d areas.
  """