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
#include "rbfwarper.h"
#include "registrationparams.h"

using namespace cv;

int main(int argc, char *argv[]) {
  registrationParams params;
  if (!params.parse(argc, argv))
    return 1;

  Magick::InitializeMagick(NULL);

  fprintf(stderr, "%ld input files listed\n", params.files.size());

  vector<Point> globalShifts;
  Rect crop(Point(0, 0), Size(0, 0));

  if (params.prereg) {
    Mat globalRefimg(grayReader().read(params.prereg_img));
    if (params.prereg_maxmove == 0) {
      params.prereg_maxmove = std::min(globalRefimg.rows, globalRefimg.cols)/2;
    }
    globalShifts = getGlobalShifts(params.files, globalRefimg, params.prereg_maxmove, true);
        crop = optimalCrop(globalShifts, globalRefimg.size());
  }

  Mat refimg;
  meanimg(params.files, crop, globalShifts, true).convertTo(refimg, CV_32F);
  cvtColor(refimg, refimg, CV_BGR2GRAY);

  const unsigned int xydiff = params.boxsize/2;
  auto patches = selectPointsHex(refimg, params.boxsize, xydiff, params.val_threshold, params.surf_threshold);
  fprintf(stderr, "%ld valid patches\n", patches.size());
  auto areas = createSearchAreas(patches, refimg.size(), params.maxmove);
  rbfWarper rbf(patches, refimg.size(), xydiff/2);

  Mat3f finalsum(Mat3f::zeros(refimg.size()));
  int progress = 0;
  #pragma omp parallel
  {
    Mat3f localsum(Mat3f::zeros(refimg.size()));
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)params.files.size(); ifile++) {
      #pragma omp critical
      fprintf(stderr, "\r\033[K%d/%ld", ++progress, params.files.size());

      Mat imgcolor;
      magickImread(params.files.at(ifile).c_str()).convertTo(imgcolor, CV_32F);
      if (params.prereg)
        imgcolor = imgcolor(crop + globalShifts.at(ifile));
      Mat1f img;
      cvtColor(imgcolor, img, CV_BGR2GRAY);
      Mat1f shifts(findShifts(img, patches, areas));
      Mat imremap(rbf.warp(imgcolor, shifts));
      localsum += imremap;
    }
    #pragma omp critical
    finalsum += localsum;
  }
  fprintf(stderr, "\n");

  double minval, maxval;
  minMaxLoc(finalsum, &minval, &maxval);
  finalsum = (finalsum - minval)/(maxval - minval) * ((1<<16)-1);
  Mat3w imgout;
  finalsum.convertTo(imgout, CV_16UC3);
  imwrite(params.output_file, imgout);

  return 0;
}
