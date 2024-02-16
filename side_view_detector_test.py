"""Library to test side_view_detector.py
.
Run via: 'python3 -m unittest side_view_detector_test' from console.
"""

from cgitb import small
import unittest  # Guide at: https://docs.python.org/3/library/unittest.html

import side_view_detector

_OUTPUT_PATH = "/Users/ryankhamneian/Desktop/Later Load Wind Foce Calculator/gowrylakshmi_code_1/ryans_test_outputs"

class SideViewDetectorTest(unittest.TestCase):
  """Class to test accuracy of side view detector."""
  def verify_generated_results(self, expected_results: side_view_detector.areas,
                               generated_results: side_view_detector.areas,
                               error_percentage: float) -> bool:
    """Function to ensure side views are within error_percentage of expected."""
    if expected_results.a_area.keys() != generated_results.a_area.keys():
      print("Failed: A keys not equal")
      return False
    if expected_results.b_area.keys() != generated_results.b_area.keys():
      print("Failed: B keys not equal")
      return False
    if expected_results.c_area.keys() != generated_results.c_area.keys():
      print("Failed: C keys not equal")
      return False
    if expected_results.d_area.keys() != generated_results.d_area.keys():
      print("Failed: D keys not equal")
      return False

    for key, value in expected_results.a_area.items():
      # print("Percentage = " + str(abs(value - generated_results.a_area[key]) / value))
      if abs(value - generated_results.a_area[key]) / value > error_percentage:
        print("Failed: a_areas not equal")
        return False
    for key, value in expected_results.b_area.items():
      if abs(value - generated_results.b_area[key]) / value > error_percentage:
        print("Failed: b_areas not equal")
        return False
    for key, value in expected_results.c_area.items():
      if abs(value - generated_results.c_area[key]) / value > error_percentage:
        print("Failed: c_areas not equal")
        return False
    for key, value in expected_results.d_area.items():
      print("Failed: d_areas not equal")
      if abs(value - generated_results.d_area[key]) / value > error_percentage:
        return False

    return True

  def test_left_1_x(self):
    image_path = "/Users/ryankhamneian/Desktop/Later Load Wind Foce Calculator/reviewed_gowrylakshmi_code_1/Side Elevations/left_1.png"
    image_length = 44.2

    generated_results = side_view_detector.get_horizontal_pressure_areas_on_x(
      image_path=image_path,
      image_length=image_length,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        1: 48,
        2: 108
      },
      b_area={
        1: 76,
        2: 0
      },
      c_area={
        1: 72,
        2: 162
      },
      d_area={
        1: 144,
      },
    )

    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))

  def test_left_2_x(self):
    image_path = "/Users/ryankhamneian/Desktop/Later Load Wind Foce Calculator/reviewed_gowrylakshmi_code_1/Side Elevations/left_2.png"
    image_length = 64.32

    generated_results = side_view_detector.get_horizontal_pressure_areas_on_x(
      image_path=image_path,
      image_length=image_length,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        1: 97,
        2: 220
      },
      b_area={
        1: 121,
        2: 0
      },
      c_area={
        1: 232,
        2: 330
      },
      d_area={
        1: 144,
      },
    )

    """
    print("\n\nExpected results 2: " + str(expected_results))
    print("\nGenerated results 2: " + str(generated_results))
    print("\nGenerated A Keys: " + str(generated_results.a_area.keys()))
    print("Generated B Keys: " + str(generated_results.b_area.keys()))
    print("Generated C Keys: " + str(generated_results.c_area.keys()))
    print("Generated D Keys: " + str(generated_results.d_area.keys()))

    print("\nExpected A Keys: " + str(expected_results.a_area.keys()))
    print("Expected B Keys: " + str(expected_results.b_area.keys()))
    print("Expected C Keys: " + str(expected_results.c_area.keys()))
    print("Expected D Keys: " + str(expected_results.d_area.keys()))
    """

    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))