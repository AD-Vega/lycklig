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

    if (context.boxsize.valid() && !params.boxsize_override)
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
    if (params.prereg == registrationParams::preregType::FirstImage)
      params.prereg_img = context.images().at(0).filename;
    else if (params.prereg == registrationParams::preregType::MiddleImage)
    {
      // Select the middle image if the number of images is odd or the image
      // just before the middle if their number is even.
      int middle = (context.images().size() + 1)/ 2 - 1;
      params.prereg_img = context.images().at(middle).filename;
    }

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

  // reference image
  Mat rawRef;
  if (params.stage_refimg || params.only_refimg ||
      (need_refimg && !context.refimg.valid())) {
    std::cerr << "Creating a stacked reference image\n";
    // This creates a color image. See below for implications.
    rawRef = meanimg(context, true);
  }

  if (params.only_refimg) {
    std::cerr << "Saving quick stack into '"
              << params.output_file << "'\n";

    Mat outputImage = rawRef;
    if (context.commonRectangle.valid() && params.crop)
    {
      // save only the region that is common to all input images
      outputImage = rawRef(context.commonRectangle());
    }
    // This saves the color image.
    imwrite(params.output_file, normalizeTo16Bits(outputImage));
  }

  // From now on, we will only store a black&white version of the reference
  // image. Pushing a new reference image into the registration context
  // means invalidating any further data (registration points, lucky imaging
  // shifts), so we will only do that if
  //   a) the creation of a new reference image was explicitly requested
  //   b) we currently don't have one, but need it in further stages
  // Note that params.only_refimg does not imply any of these!
  if (params.stage_refimg || (need_refimg && !context.refimg.valid())) {
    // Save the black&white reference image to context.
    Mat refimg;
    if (rawRef.channels() > 1)
      cvtColor(rawRef, refimg, CV_BGR2GRAY);
    else
      refimg = rawRef;
    context.refimg(refimg);

    // Changing the reference image invalidates lucky imaging registration
    // points.
    std::cerr << "New reference image created\n";
    context.clearPatchesEtc();
  }

  // Check whether we need to override context.boxsize() with a value from
  // the command line. If there is a conflict, we invalidate any further data
  // (registration points, lucky imaging shifts).
  if (need_patches && params.boxsize_override && context.boxsize.valid() &&
      params.boxsize != context.boxsize()) {
    std::cerr << "New boxsize specified on the command line\n";
    context.clearPatchesEtc();
  }

  // Where to create the registration points.
  Rect patchCreationArea = context.refimgRectangle();
  if (params.crop && context.commonRectangle.valid()) {
    // The reference image is usually larger than commonRectangle and we can
    // expand the patch creation area so that the reference points are placed
    // right on the edge of commonRectangle.
    const Rect cr = context.commonRectangle();
    const Point halfbox(params.boxsize/2, params.boxsize/2);
    const Rect expandedSearch(cr.tl() - halfbox, cr.br() + halfbox);
    // But do cautiously trim the expanded rectangle so that it fits within
    // the reference image.
    patchCreationArea = expandedSearch & context.refimgRectangle();
  }

  // If we have the registration points already, check that they were created
  // with the same crop option.
  if (need_patches && context.patches.valid() &&
      patchCreationArea != context.patches().patchCreationArea)
  {
    std::cerr << "Existing registration points were created with different crop settings\n";
    context.clearPatchesEtc();
  }

  // lucky imaging registration points
  if (params.stage_patches || (need_patches && !context.patches.valid())) {
    context.boxsize(params.boxsize);

    std::cerr << "Lucky imaging: creating registration patches\n";
    auto patches = selectPointsHex(params, context, patchCreationArea);
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
    else if (params.stage_stack && context.shifts.valid())
      std::cerr << "Stacking images (using data from lucky imaging)\n";
    else if (params.stage_stack)
      std::cerr << "Stacking images (no lucky imaging)\n";

    Mat finalsum = lucky(params, context, true);
    // Only save the result if there is something to save.
    if (params.stage_stack) {
      std::cerr << "Saving output to '" << params.output_file << "'\n";
      imwrite(params.output_file, normalizeTo16Bits(finalsum));
    }
  }

  if (!params.save_state_file.empty()) {
    std::cerr << "Saving state to '" << params.save_state_file << "'\n";
    FileStorage saveStateFS(params.save_state_file, FileStorage::WRITE);
    context.write(saveStateFS);
  }

  return 0;
}
