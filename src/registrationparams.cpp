/*
 *    lycklig, image processing for lucky imaging.
 *    Copyright (C) 2013, 2014 Andrej Lajovic <andrej.lajovic@ad-vega.si>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tclap/CmdLine.h>
#include "registrationparams.h"

class naturalNumberConstraint : public TCLAP::Constraint<unsigned int>
{
public:
  std::string description() const
    { return "integer >= 1"; }
  std::string shortID() const
    { return "N"; }
  bool check(const unsigned int& value) const
    { return (value >= 1); }
} nnConstraint;


template <typename num_t>
std::string defval(num_t d)
{
  return "(default " + std::to_string(d) + ")";
}


bool registrationParams::parse(const int argc, const char* argv[])
{
  try
  {
    TCLAP::CmdLine cmd("Registration of planetary images");

    // pre-registration
    TCLAP::ValueArg<std::string> arg_prereg_img(
      "p", "prereg-img", "Preregister using this image as the reference.", false, "", "filename");
    cmd.add(arg_prereg_img);
    TCLAP::SwitchArg arg_prereg_on_first(
      "1", "prereg-on-first", "Preregister using the first image as the reference.", false);
    cmd.add(arg_prereg_on_first);
    TCLAP::SwitchArg arg_prereg_on_middle(
      "2", "prereg-on-middle", "Preregister using the middle image as the reference.", false);
    cmd.add(arg_prereg_on_middle);
    TCLAP::ValueArg<unsigned int> arg_prereg_maxmove(
      "x", "prereg-maxmove", "Maximum displacement in pre-registering. Zero means half "
                             "of the images' smallest size; this is also the default.",
                             false, prereg_maxmove, "pixels");
    cmd.add(arg_prereg_maxmove);

    // reference image
    TCLAP::SwitchArg arg_refimg(
      "r", "refimg", "Create a reference image to be used as a template for dedistortion.", stage_refimg);
    cmd.add(arg_refimg);
    TCLAP::SwitchArg arg_only_refimg(
      "n", "only-refimg", "Only roughly stack (possibly pre-registered) images (implies --refimg).", only_refimg);
    cmd.add(arg_only_refimg);
    TCLAP::SwitchArg arg_crop(
      "c", "crop", "Crop the image to the area common to all input images "
                   "(only effective with pre-registration; a no-op otherwise).", crop);
    cmd.add(arg_crop);

    // registration points
    TCLAP::SwitchArg arg_patches(
      "a", "patches", "Create registration points for dedistortion.", stage_patches);
    cmd.add(arg_patches);
    TCLAP::ValueArg<unsigned int> arg_boxsize(
      "b", "boxsize", "Box size " + defval(boxsize), false, boxsize, "pixels");
    cmd.add(arg_boxsize);

    // dedistortion
    TCLAP::SwitchArg arg_dedistortion(
      "d", "dedistort", "Dedistortion.", stage_dedistort);
    cmd.add(arg_dedistortion);
    TCLAP::ValueArg<unsigned int> arg_maxmove(
      "m", "maxmove", "Maximum displacement in dedistortion " + defval(maxmove), false, maxmove, "pixels");
    cmd.add(arg_maxmove);

    // interpolation + stacking
    TCLAP::SwitchArg arg_stack(
      "t", "stack", "Stack (sum) the resulting images", stage_stack);
    cmd.add(arg_stack);
    TCLAP::ValueArg<unsigned int> arg_supersampling(
      "s", "super", "Supersampling " + defval(supersampling), false, supersampling, "N");
    cmd.add(arg_supersampling);

    // input options
    TCLAP::ValueArg<std::string> arg_read_state(
      "i", "read-state", "Continue processing from a saved state", false, "", "filename.yml");
    cmd.add(arg_read_state);
    TCLAP::UnlabeledMultiArg<std::string> arg_files(
      "files", "Image files to process", false, "files");
    cmd.add(arg_files);

    // output options
    TCLAP::ValueArg<std::string> arg_save_state(
      "w", "save-state", "Save the registration state into a file", false, "", "filename.yml");
    cmd.add(arg_save_state);
    TCLAP::ValueArg<std::string> arg_output_file(
      "o", "output", "Output file", false, "", "filename");
    cmd.add(arg_output_file);

    cmd.parse(argc, argv);

    // stages
    if (arg_prereg_img.isSet() +
        arg_prereg_on_first.isSet() +
        arg_prereg_on_middle.isSet() > 1) {
      std::cerr << "ERROR: arguments --prereg-img, --prereg-on-first and\n"
                << "       --prereg-on-middle are mutually exclusive!" << std::endl;
      return false;
    }
    if (arg_prereg_img.isSet()) {
      prereg = preregType::ExplicitImage;
      prereg_img = arg_prereg_img.getValue();
    }
    else if (arg_prereg_on_first.isSet())
      prereg = preregType::FirstImage;
    else if (arg_prereg_on_middle.isSet())
      prereg = preregType::MiddleImage;

    if (prereg != preregType::None)
      stage_prereg = true;
    stage_refimg = arg_refimg.isSet();
    stage_patches = arg_patches.isSet();
    stage_dedistort = arg_dedistortion.isSet();
    stage_stack = arg_stack.isSet();

    // options
    only_refimg = arg_only_refimg.isSet();
    if (only_refimg && stage_stack) {
      std::cerr << "ERROR: --only-refimg and --stack can not be enabled at the same time." << std::endl;
      return false;
    }

    prereg_maxmove = arg_prereg_maxmove.getValue();
    boxsize_override = arg_boxsize.isSet();
    boxsize = arg_boxsize.getValue();
    crop = arg_crop.isSet();
    maxmove = arg_maxmove.getValue();
    supersampling = arg_supersampling.getValue();

    if (arg_read_state.isSet() && arg_files.isSet()) {
      std::cerr << "ERROR: you can either use --read-state OR list input files." << std::endl;
      return false;
    }
    if (arg_read_state.isSet()) {
      read_state_file = arg_read_state.getValue();
      if (read_state_file.length() >= 4 &&
          std::string(read_state_file.cend()-4, read_state_file.cend()) != ".yml") {
        std::cerr << "ERROR: --read-state requires a file name ending in '.yml'\n"
                     "       (sorry - an OpenCV peculiarity; can't do much about that)\n";
        return false;
      }
    }
    else {
      files = arg_files.getValue();
      if (files.size() == 0) {
        std::cerr << "ERROR: No input files given\n";
        return false;
      }
    }

    if (arg_save_state.isSet()) {
      save_state_file = arg_save_state.getValue();
      if (save_state_file.length() >= 4 &&
          std::string(save_state_file.cend()-4, save_state_file.cend()) != ".yml") {
        std::cerr << "ERROR: --save-state requires a file name ending in '.yml'\n"
                     "       (sorry - an OpenCV peculiarity; can't do much about that)\n";
        return false;
      }
    }

    if (arg_output_file.isSet()) {
      if (! (only_refimg || stage_stack)) {
        std::cerr << "ERROR: --output file given but no image-producing stages are enabled" << std::endl;
        return false;
      }
      output_file = arg_output_file.getValue();
    }
    else {
      if (only_refimg || stage_stack) {
        std::cerr << "ERROR: stacking enabled, but no --output given.\n"
                     "       Refusing to discard the result.\n";
        return false;
      }
      else if (!arg_save_state.isSet()) {
        std::cerr << "ERROR: no destination file specified with --save-state.\n"
                     "       Refusing to discard data.\n";
        return false;
      }
    }
  }
  catch (TCLAP::ArgException &e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return false;
  }
  return true;
}
