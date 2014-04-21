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
    TCLAP::ValueArg<std::string> arg_prereg_img(
      "p", "prereg-img", "Preregister using this image as the reference.", false, "", "filename");
    cmd.add(arg_prereg_img);
    TCLAP::SwitchArg arg_prereg_first(
      "1", "prereg-first", "Preregister using the first image as the reference.", false);
    cmd.add(arg_prereg_first);
    TCLAP::ValueArg<unsigned int> arg_prereg_maxmove(
      "x", "prereg-maxmove", "Maximum displacement in pre-registering. Zero means half "
                             "of the images' smallest size; this is also the default.",
                             false, prereg_maxmove, "pixels");
    cmd.add(arg_prereg_maxmove);
    TCLAP::SwitchArg arg_only_stack(
      "n", "only-stack", "Don't do lucky imaging: only stack (possibly pre-registered) images.", false);
    cmd.add(arg_only_stack);
    TCLAP::ValueArg<unsigned int> arg_boxsize(
      "b", "boxsize", "Box size " + defval(boxsize), false, boxsize, "pixels");
    cmd.add(arg_boxsize);
    TCLAP::ValueArg<unsigned int> arg_maxmove(
      "m", "maxmove", "Maximum displacement " + defval(maxmove), false, maxmove, "pixels");
    cmd.add(arg_maxmove);
    TCLAP::ValueArg<std::string> arg_output_file(
      "o", "output", "Output file", true, "", "filename");
    cmd.add(arg_output_file);
    TCLAP::UnlabeledMultiArg<std::string> arg_files(
      "files", "Image files to process", true, "files");
    cmd.add(arg_files);

    cmd.parse(argc, argv);

    boxsize = arg_boxsize.getValue();
    maxmove = arg_maxmove.getValue();
    output_file = arg_output_file.getValue();
    files = arg_files.getValue();

    if (arg_prereg_img.isSet() + arg_prereg_first.isSet() > 1) {
      std::cerr << "PARSE ERROR: arguments --prereg-img and --prereg-first\n"
                << "             are mutually exclusive!" << std::endl;
      return false;
    }
    if (arg_prereg_img.isSet()) {
      prereg = true;
      prereg_img = arg_prereg_img.getValue();
    }
    else if (arg_prereg_first.isSet()) {
      prereg = true;
      prereg_img = files.at(0);
    }
    if (arg_only_stack.isSet())
      only_stack = true;
  }
  catch (TCLAP::ArgException &e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return false;
  }
  return true;
}
