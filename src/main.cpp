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

#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include "imageops.h"
#include "lucky.h"
#include "globalregistrator.h"
#include "rbfwarper.h"
#include "registrationparams.h"
#include "registrationcontext.h"

using namespace cv;

int main(const int argc, const char *argv[]) {
  // Parse command line parameters.
  registrationParams params;
  if (!params.parse(argc, argv))
    return 1;

  // This should be called in order for OpenMP parallelization to work with
  // ImageMagick.
  Magick::InitializeMagick(NULL);

  registrationContext context;

  // Resolve stage dependencies.
  bool need_patches = params.stage_patches || params.stage_lucky;
  bool need_refimg = params.stage_refimg || need_patches;

  // Load a state file if one was supplied.
  if (!params.read_state_file.empty()) {
    std::cerr << "Reading state from '" << params.read_state_file << "':\n";
    FileStorage readStateFS(params.read_state_file, FileStorage::READ);
    context = registrationContext(readStateFS);
    context.printReport();
    std::cerr << std::endl;

    if (context.boxsizeValid() && !params.boxsize_override)
      params.boxsize = context.boxsize();
  }
  else {
    // No state file - we are starting from scratch. Initialize registration
    // context from command line parameters.
    std::vector<inputImage> images;
    for (auto& file : params.files)
      images.push_back(inputImage(file));
    context.images(images);
    std::cerr << context.images().size() << " input files listed on command line\n";
    auto sampleFile = images.at(0).filename;
    std::cerr << "Probing '" << sampleFile << "' for size... ";
    Mat sample = magickImread(sampleFile);
    context.imagesize(sample.size());
    std::cerr << context.imagesize().width << "x"
              << context.imagesize().height << "\n";
  }

  // preregistration stage
  if (params.stage_prereg) {
    if (params.prereg_img.empty())
      params.prereg_img = context.images().at(0).filename;
    Mat globalRefimg(grayReader().read(params.prereg_img));
    if (params.prereg_maxmove == 0) {
      params.prereg_maxmove = std::min(globalRefimg.rows, globalRefimg.cols)/2;
    }
    std::cerr << "Pre-registering on reference '" << params.prereg_img << "'\n";
    globalRegistrator::getGlobalShifts(params, context, globalRefimg, true);

    // New global shifts invalidate any further data in the context.
    std::cerr << "New pre-registration data obtained\n";
    context.clearRefimgEtc();
  }

  // Check for consistency of crop settings and ensure that we have a properly
  // sized reference image.
  if (context.refimgValid() && context.commonRectangleValid()) {
    if ((params.crop && context.refimg().size() != context.commonRectangle().size()) ||
        (!params.crop && context.refimg().size() != context.imagesize())) {
      std::cerr << "Reference image size not consistent with crop options\n";
      context.clearRefimgEtc();
    }
  }

  // reference image
  Mat rawRef;
  if (params.stage_refimg || params.only_refimg ||
      (need_refimg && !context.refimgValid())) {
    std::cerr << "Creating a stacked reference image\n";
    // This creates a color image. See below for implications.
    rawRef = meanimg(params, context, true);
  }
  if (params.only_refimg) {
    std::cerr << "Saving quick stack into '"
              << params.output_file << "'\n";
    // This saves the color image.
    imwrite(params.output_file, normalizeTo16Bits(rawRef));
  }
  // From now on, we will only store a black&white version of the reference
  // image. Pushing a new reference image into the registration context
  // means invalidating any further data (registration points, lucky imaging
  // shifts), so we will only do that if
  //   a) the creation of a new reference image was explicitly requested
  //   b) we currently don't have one, but need it in further stages
  // Note that params.only_refimg does not imply any of these!
  if (params.stage_refimg || (need_refimg && !context.refimgValid())) {
    // Save the black&white reference image to context.
    Mat refimg;
    cvtColor(rawRef, refimg, CV_BGR2GRAY);
    context.refimg(refimg);

    // Changing the reference image invalidates lucky imaging registration
    // points.
    std::cerr << "New reference image created\n";
    context.clearPatchesEtc();
  }

  // Check whether we need to override context.boxsize() with a value from
  // the command line. If there is a conflict, we invalidate any further data
  // (registration points, lucky imaging shifts).
  if (need_patches && params.boxsize_override && context.boxsizeValid() &&
      params.boxsize != context.boxsize()) {
    std::cerr << "New boxsize specified on the command line\n";
    context.clearPatchesEtc();
  }

  // lucky imaging registration points
  if (params.stage_patches || (need_patches && !context.patchesValid())) {
    context.boxsize(params.boxsize);
    std::cerr << "Lucky imaging: creating registration patches\n";
    auto patches = selectPointsHex(params, context);
    patches = filterPatchesByQuality(patches, context.refimg());
    context.patches(patches);
    std::cerr << context.patches().size() << " valid patches\n";

    // Changing the registration points invalidates lucky imaging shifts.
    context.clearShiftsEtc();
  }

  if (params.stage_lucky || params.stage_stack) {
    if (params.stage_lucky && params.stage_stack)
      std::cerr << "Lucky imaging: registration, warping and stacking\n";
    else if (params.stage_lucky)
      std::cerr << "Lucky imaging: registration\n";
    else if (params.stage_stack && context.shiftsValid())
      std::cerr << "Stacking images (using data from lucky imaging)\n";
    else if (params.stage_stack)
      std::cerr << "Stacking images (no lucky imaging)\n";

    Mat finalsum = lucky(params, context, true);
    // Only save the result if there is something to save.
    if (params.stage_stack)
      std::cerr << "Saving output to '" << params.output_file << "'\n";
      imwrite(params.output_file, normalizeTo16Bits(finalsum));
  }

  if (!params.save_state_file.empty()) {
    std::cerr << "Saving state to '" << params.save_state_file << "'\n";
    FileStorage saveStateFS(params.save_state_file, FileStorage::WRITE);
    context.write(saveStateFS);
  }

  return 0;
}
