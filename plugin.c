#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gtk/gtk.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>
#include <libgimpwidgets/gimpwidgets.h>

#include <fftw3.h>

/** Defines ******************************************************************/

#define CLAMPED(x,l,r) (((x)<(l))?(l):(((x)>(r))?(r):(x)))

#define PLUG_IN_NAME "plug_in_addam_test"
#define PLUG_IN_BINARY "gimp-test"
#define PLUG_IN_AUTHOR "Addam Dominec"
#define PLUG_IN_VERSION "Aug 2011, 0.4.1"

#define WAVELET_DEPTH 6
#define GRAPH_WIDTH 200
#define GRAPH_HEIGHT 200

#define USER_POINT_COUNT 15
#define GRAPH_HOTSPOT 3

typedef struct _Curve {
	GdkPoint points[GRAPH_WIDTH];
	GdkPoint user_points[USER_POINT_COUNT];
	guchar count;
} Curve;

typedef struct PluginData {
  GimpDrawable      *drawable;
  GimpPixelRgn       region;

	gint              img_width, img_height, img_offset_x, img_offset_y;
	gint              img_bpp;
	fftwf_complex   **image_freq; // array of pointers to image for each channel, frequency domain
	short            *image_wavelet; // an array of wavelet images: y->x->scale
	float           **image;  // same as above, image domain (fixme: remove?)
	guchar           *img_pixels; // array used for acquiring data from the drawable
	fftwf_plan        plan;

	Curve             curve_user, curve_fft;
	gboolean          curve_user_changed;
	GtkWidget        *graph;
	float             histogram[GRAPH_WIDTH];
	int               point_grabbed; //FIXME: patří ke křivce
	GdkPixmap        *graph_pixmap;
	
	GtkWidget        *preview;
	gboolean          do_preview_hd;
} PluginData;

void curve_copy(Curve *src, Curve *dest);
float curve_get_value(float x, Curve *c); // interpolate the curve at an arbitrary point in range [0; 1]
float index_to_dist(int index, int width, int height); // get wavelenght from index in FFT array
float graph_to_dist(float x);
float dist_to_graph(float dist); // map pixel distance to graph x value
int value_to_graph(float val);
gint preview_hd(GtkWidget *preview, PluginData *pd);
/** Plugin interface *********************************************************/

static void query(void);
static void run (const gchar      *name,
                 gint              nparams,
                 const GimpParam  *param,
                 gint             *nreturn_vals,
                 GimpParam       **return_vals);

GimpPlugInInfo PLUG_IN_INFO = {NULL, NULL, query, run};

void fft_prepare(PluginData *pd)
{
	gint         w = pd->img_width, h = pd->img_height;
	gint         img_bpp = pd->img_bpp, cur_bpp;
	int          x, y;
	float      **image;
	guchar      *img_pixels;
	float        norm;
	image = pd->image = (float**) malloc(sizeof(float*) * img_bpp);
	pd->image_freq = (fftwf_complex**) malloc(sizeof(fftwf_complex*) * img_bpp);
  img_pixels = pd->img_pixels = g_new (guchar, w * h * img_bpp);
  //allocate an array for each channel
  for (cur_bpp=0; cur_bpp<img_bpp; cur_bpp ++){
	  image[cur_bpp] = (float*) fftwf_malloc(sizeof(float) * w * h);
		pd->image_freq[cur_bpp] = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (w/2+1) * h);
	}
	printf("Image data occupies %lu MB.\n", (sizeof(float) * w * h * img_bpp) >> 20);
	printf("Frequency data occupies %lu MB.\n", (sizeof(fftwf_complex) * (w/2+1) * h * img_bpp) >> 20);
	
	// forward plan
	fftwf_plan plan = fftwf_plan_dft_r2c_2d(pd->img_height, pd->img_width, *image, *pd->image_freq, FFTW_ESTIMATE);
	// inverse plan (to be reused)
	pd->plan = fftwf_plan_dft_c2r_2d(pd->img_height, pd->img_width, *pd->image_freq, *image, FFTW_ESTIMATE);

	// set image region to reading mode
	gimp_pixel_rgn_init (&pd->region, pd->drawable, pd->img_offset_x, pd->img_offset_y, w, h, FALSE, FALSE);
	gimp_pixel_rgn_get_rect(&pd->region, img_pixels, pd->img_offset_x, pd->img_offset_y, w, h);
	
	// execute forward FFT once
	int pw = w/2+1; // physical width
	float diagonal = sqrt(h*h + w*w)/2.0;
	norm = 1.0/(w*h);
	for(cur_bpp=0; cur_bpp<img_bpp; cur_bpp++)
	{
		// convert one color channel to float[]
		for(x=0; x < w; x ++)
		{
			for(y=0; y < h; y ++)
			{
				 image[cur_bpp][y*w + x] =  (float) img_pixels[(y*w + x)*img_bpp + cur_bpp] * norm;
			}
		}
		// transform the channel
		fftwf_execute_dft_r2c(plan, image[cur_bpp], pd->image_freq[cur_bpp]);
	}
	fftwf_destroy_plan(plan);
}
void fft_apply(PluginData *pd)
{
	int w = pd->img_width, h = pd->img_height,
		pw = w/2+1; // physical width
	fftwf_complex *multiplied = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * pw * h);
	float diagonal = sqrt(h*h + w*w)/2.0;
	// save current state of the curve
	curve_copy(&pd->curve_user, &pd->curve_fft);
	for(int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp++)
	{
		//skip DC value
		multiplied[0][0] = pd->image_freq[cur_bpp][0][0];
		multiplied[0][1] = pd->image_freq[cur_bpp][0][1];
		// apply convolution
		for (int i=1; i < pw*h; i++){
			float dist = index_to_dist(i, pw, h);
			float coef = curve_get_value(dist_to_graph(dist), &pd->curve_fft);
			multiplied[i][0] = pd->image_freq[cur_bpp][i][0] * coef;
			multiplied[i][1] = pd->image_freq[cur_bpp][i][1] * coef;
		}
		// apply inverse FFT
		fftwf_execute_dft_c2r(pd->plan, multiplied, pd->image[cur_bpp]);
		// pack results for GIMP
		for(int x=0; x < w; x ++)
		{
			for(int y=0; y < h; y ++)
			{
				float v = pd->image[cur_bpp][y*w + x];
				pd->img_pixels[(y*w + x) * pd->img_bpp+cur_bpp] = CLAMPED(v,0,255);
			}
		}
	}
	fftwf_free(multiplied);
}
void fft_destroy(PluginData *pd)
{
	fftwf_destroy_plan(pd->plan);
	for (int i=0; i<pd->img_bpp; i++){
		fftwf_free(pd->image[i]);
		fftwf_free(pd->image_freq[i]);
	}
	free(pd->image);
	free(pd->image_freq);
}
int scale_to_dist(int scale, int diagonal){
	if (scale >= WAVELET_DEPTH-1 || 1UL<<WAVELET_DEPTH-1 > diagonal)
		return diagonal;
	else
		return 1UL<<(scale+2);
}
void wavelet_prepare(PluginData *pd){
	int w = pd->img_width, h = pd->img_height,
		pw = w/2+1; // physical width
	fftwf_complex *multiplied = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * pw * h);
	float *image_temp = (float*)fftwf_malloc(sizeof(float) * w * h);
	float diagonal = sqrt(h*h + w*w)/2;
	pd->image_wavelet = (short*)fftwf_malloc(WAVELET_DEPTH * w * h * sizeof(short));
	printf("Wavelet layers occupy %lu MB.\n", (WAVELET_DEPTH * w * h * sizeof(short)) >> 20);
	
	int lower = 0, peak = 1, upper = scale_to_dist(1, diagonal);
	for (int scale = 0; scale < WAVELET_DEPTH; scale ++)
	{
		float above = upper-peak, below = peak-lower;
		for (int i=0; i < pw*h; i++){
			multiplied[i][0] = multiplied[i][1] = 0.0;
		}
		for (int i=0; i < pw*h; i++)
		{
			//fixme: simplify
			float dist = index_to_dist(i, pw, h);
			if (dist <= upper){
				if (dist > lower){
					if (dist > peak){
						for(int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
						{
							multiplied[i][0] += pd->image_freq[cur_bpp][i][0];
							multiplied[i][1] += pd->image_freq[cur_bpp][i][1];
						}
						float coef = (1.0 - (dist-peak)/above) / pd->img_bpp;
						multiplied[i][0] *= coef;
						multiplied[i][1] *= coef;
					}
					else {
						for(int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
						{
							multiplied[i][0] += pd->image_freq[cur_bpp][i][0];// - multiplied[i][0];
							multiplied[i][1] += pd->image_freq[cur_bpp][i][1];// - multiplied[i][1];
						}
						float coef = (1.0 - (peak-dist)/below) / pd->img_bpp;
						multiplied[i][0] *= coef;
						multiplied[i][1] *= coef;
					}
				}
				else {
					multiplied[i][0] = multiplied[i][1] = 0.0;
				}
			}
			else {
				multiplied[i][0] = multiplied[i][1] = 0.0;
			}
		}
		// apply inverse FFT
		fftwf_execute_dft_c2r(pd->plan, multiplied, image_temp);
		for (int i=0; i < w*h; i++)
		{
			pd->image_wavelet[i*WAVELET_DEPTH + scale] = image_temp[i];//CLAMPED(image_temp[i], -128, 127);
		}
		lower = peak;
		peak = upper;
		upper = scale_to_dist(scale+2, diagonal);
	}
	fftwf_free(multiplied);
	fftwf_free(image_temp);
}
void wavelet_apply(PluginData *pd, int out_x, int out_y, int out_w, int out_h){
	int w = pd->img_width, h = pd->img_height;
	if (pd->curve_user_changed)
	{
		// estimate needed coefficient for each wavelet layer
		// (TODO: integrate)
		float coef[WAVELET_DEPTH];
		float diagonal = sqrt(h*h + w*w)/2;
		for (int scale=0; scale<WAVELET_DEPTH; scale++)
		{
			float x = dist_to_graph(scale_to_dist(scale, diagonal));
			coef[scale] = curve_get_value(x, &pd->curve_user) - curve_get_value(x, &pd->curve_fft);
		}
		// combine wavelet layers
		for (int y=0; y<out_h; y++)
		{
			for (int x=0; x<out_w; x++)
			{
				short *pixel_wavelets = pd->image_wavelet + WAVELET_DEPTH*((y+out_y)*w +(x+out_x));
				float diff = 0; // needed value of the pixel
				for (int scale=0; scale<WAVELET_DEPTH; scale ++)
				{
					diff += pixel_wavelets[scale] * coef[scale];
				}
				if (diff < 0)
				{
					// darken the pixel: multiply all channels
					float value = 0; // current value of the pixel
					for (int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
					{
						value += pd->image[cur_bpp][(y+out_y)*w+(x+out_x)];
					}
					value = value/pd->img_bpp;
					for (int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
					{
						pd->img_pixels[(y*out_w+x)*pd->img_bpp + cur_bpp] = CLAMPED(pd->image[cur_bpp][(y+out_y)*w+(x+out_x)] * (1 + diff/value), 0, 255);
					}
				}
				else
				{
					// brighten the pixel: add value to all channels
					for (int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
					{
						pd->img_pixels[(y*out_w+x)*pd->img_bpp + cur_bpp] = CLAMPED(pd->image[cur_bpp][(y+out_y)*w+(x+out_x)] + diff, 0, 255);
					}
				}
			}
		}
	}
	else
	{
		// direct copy
		for (int y=0; y<out_h; y++)
		{
			for (int x=0; x<out_w; x++)
			{
				for (int cur_bpp=0; cur_bpp < pd->img_bpp; cur_bpp ++)
				{
					pd->img_pixels[(y*out_w+x)*pd->img_bpp + cur_bpp] = CLAMPED(pd->image[cur_bpp][(y+out_y)*w+(x+out_x)], 0, 255);
				}
			}
		}
	}
}
void wavelet_destroy(PluginData *pd){
	fftwf_free(pd->image_wavelet);
}
// Generate a histogram from FFT data
void histogram_generate(PluginData *pd){
	int pw = pd->img_width/2+1, h = pd->img_height;
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		pd->histogram[i] = 0;
	}
	for(int cur_bpp=0; cur_bpp<pd->img_bpp; cur_bpp++)
	{
		// add value to histogram
		for(int i=0; i<pw*h; i++)
		{
			float *pixel = (float*)(pd->image_freq[cur_bpp] + i);
			float val = sqrt(pixel[0]*pixel[0] + pixel[1]*pixel[1]);
			float dist = index_to_dist(i, pw, h);
			pd->histogram[(unsigned)(GRAPH_WIDTH*CLAMPED(dist_to_graph(dist), 0, 1))] += val/(dist+1);
		}
	}
	// remap histogram values
	float histogram_max = 0;
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		pd->histogram[i] = log(pd->histogram[i]+1.0);
		if (pd->histogram[i] > histogram_max)
			histogram_max = pd->histogram[i];
	}
	histogram_max = 1.0/histogram_max;
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		pd->histogram[i] *= histogram_max;
	}
}
void graph_redraw(PluginData *pd)
{
	GtkStyle *graph_style = gtk_widget_get_style (pd->graph);
	// clear the pixmap
	gdk_draw_rectangle (pd->graph_pixmap, graph_style->light_gc[GTK_STATE_NORMAL],
											TRUE, 0, 0, GRAPH_WIDTH, GRAPH_HEIGHT);
	
	// histogram
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		gdk_draw_line (pd->graph_pixmap, graph_style->mid_gc[GTK_STATE_NORMAL], i, GRAPH_HEIGHT * (1.0-pd->histogram[i]), i, GRAPH_HEIGHT);
	}
	// horizontal lines
	gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], 0, GRAPH_HEIGHT-1, GRAPH_WIDTH, GRAPH_HEIGHT-1);
	for (int i = 1; i < 10; i++)
	{
		int y = value_to_graph(i/2.0);
		gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], 0, y, GRAPH_WIDTH, y);
	}
	gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], 0, 0, GRAPH_WIDTH, 0);
	// vertical lines
	for (int i = 0; i < 10; i++)
	{
		int x = dist_to_graph(i)*GRAPH_WIDTH;
		gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], x, 0, x, GRAPH_HEIGHT);
		x = dist_to_graph(10*i)*GRAPH_WIDTH;
		gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], x, 0, x, GRAPH_HEIGHT);
	}
	gdk_draw_line (pd->graph_pixmap, graph_style->dark_gc[GTK_STATE_NORMAL], GRAPH_WIDTH-1, 0, GRAPH_WIDTH-1, GRAPH_HEIGHT);
	// wavelet marks
	float diagonal = (pd->img_width*pd->img_width + pd->img_height*pd->img_height)/4;
	for (int i = 0; i < WAVELET_DEPTH; i++)
	{
		int x = CLAMPED(dist_to_graph(scale_to_dist(i, diagonal))*GRAPH_WIDTH, 0, GRAPH_WIDTH-1);
		gdk_draw_line (pd->graph_pixmap, graph_style->text_gc[GTK_STATE_NORMAL], x, GRAPH_HEIGHT/2 - 2, x, GRAPH_HEIGHT/2 + 2);
	}
	// user curve
	gdk_draw_lines (pd->graph_pixmap, graph_style->text_gc[GTK_STATE_NORMAL], pd->curve_user.points, GRAPH_WIDTH);
	// user points
	for (int i = 0; i < pd->curve_user.count; i++)
	{
		gdk_draw_arc(pd->graph_pixmap, graph_style->text_gc[GTK_STATE_NORMAL], FALSE, pd->curve_user.user_points[i].x-GRAPH_HOTSPOT-1, pd->curve_user.user_points[i].y-GRAPH_HOTSPOT-1, GRAPH_HOTSPOT*2+1, GRAPH_HOTSPOT*2+1, 0, 64*360);
	}
	gdk_draw_drawable (pd->graph->window, graph_style->text_gc[GTK_STATE_NORMAL], pd->graph_pixmap, 0, 0, 0, 0, GRAPH_WIDTH, GRAPH_HEIGHT);
}

void curve_init(Curve *c){
	c->count = 0;
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		c->points[i].x = i;
		c->points[i].y = GRAPH_HEIGHT/2;
	}
}
void curve_copy(Curve *src, Curve *dest){
	dest->count = src->count;
	for (int i=0; i<src->count; i++)
	{
		dest->user_points[i] = src->user_points[i];
	}
	for (int i=0; i<GRAPH_WIDTH; i++)
	{
		dest->points[i] = src->points[i];
	}
}

// FFT array index -> frequency magnitude (unnormalized)
float index_to_dist(int index, int width, int height)
{
	int x, y;
	x = index % width;
	y = index / width;
	if (y > height/2){
		y = y-height;
	}
	return sqrt(x*x + y*y);
}
	
// frequency magnitude (unnormalized) -> graph x (normalized)
float dist_to_graph(float dist)
{
	float point = 1 - 1/(dist+1);
	point = point * point;
	return point * point * point;
}

// graph x coordinate (normalized) -> frequency magnitude (unnormalized)
float graph_to_dist(float x)
{
	float point = pow(x, 1.0/6);
	return 1/(1-point) - 1;
}
	
// value -> graph (unnormalized)
int value_to_graph(float val)
{
	if (val<1)
		return GRAPH_HEIGHT*(1.0-val/2);
	else
		return GRAPH_HEIGHT/(2*val);
}

// graph (unnormalized) -> value
float graph_to_value(int y)
{
	if (y > GRAPH_HEIGHT/2)
		return (GRAPH_HEIGHT - y - 1) / (float)(GRAPH_HEIGHT/2.0);
	else if (y>0)
		return (GRAPH_HEIGHT/2.0) / (float) y;
	else
		return GRAPH_HEIGHT/2.0;
}

// interpolate the curve at a given point between two points with indices i1, i2 (unnormalized)
float curve_interpolate(float x, int i1, int i2, Curve *c)
{
	// linear interpolation
	float x1 = graph_to_dist(c->user_points[i1].x/(float)GRAPH_WIDTH), x2 = graph_to_dist(c->user_points[i2].x/(float)GRAPH_WIDTH);
	x = graph_to_dist(x/(float)GRAPH_WIDTH);
	return ((x - x1) * graph_to_value(c->user_points[i2].y) + (x2 - x) * graph_to_value(c->user_points[i1].y)) / (x2 - x1);
}

// get index to the first larger element (compare by coordinate y)
int gdkpoint_bisect (int item, GdkPoint *array, int count)
{
	int left = 0, right = count;
	while (left != right){
		int test = (left+right)/2;
		if (array[test].x > item)
			right = test;
		else
			left = test+1;
	}
	return left;
}

// interpolate the curve at an arbitrary point in range [0; 1]
float curve_get_value(float x, Curve *c)
{
	if (c->count == 0)
		return 1.0;// No curve - constant 1
	x = x * GRAPH_WIDTH;
	int index = gdkpoint_bisect(x, c->user_points, c->count);
	// constant extrapolation
	if (index == c->count)
		return graph_to_value(c->user_points[index-1].y);
	else if (index == 0)
		return graph_to_value(c->user_points[index].y);
	// interpolation
	else
		return curve_interpolate(x, index-1, index, c);
}

// update curve parts to the left and to the right of given point
void curve_update_part(int index, Curve *c){
	if (index > 0)
		for (int i=c->user_points[index-1].x; i<c->user_points[index].x; i++){
			c->points[i].y = value_to_graph(curve_interpolate(i, index-1, index, c));
		}
	else
		for (int i=0; i<c->user_points[index].x; i++){
			c->points[i].y = c->user_points[index].y;
		}
	if (index+1 < c->count)
		for (int i=c->user_points[index+1].x; i>=c->user_points[index].x; i--){
			c->points[i].y = value_to_graph(curve_interpolate(i, index, index+1, c));
		}
	else
		for (int i=GRAPH_WIDTH-1; i>=c->user_points[index].x; i--){
			c->points[i].y = c->user_points[index].y;
		}
}

// insert a point into the curve
int curve_add_point(int x, int y, Curve *c)
{
	// get neighbors' positions (if any)
	GdkPoint point = {x, y};
	int i, index = gdkpoint_bisect(x, c->user_points, c->count);
	for (i=c->count; i>index; i--){
		c->user_points[i] = c->user_points[i-1];
	}
	c->user_points[index] = point;
	c->count += 1;
	curve_update_part(index, c);
	return index;
}

// remove a point from the curve
void curve_remove_point(int index, Curve *c)
{
	c->count -= 1;
	int i;
	for (i=index; i < c->count; i++){
		c->user_points[i] = c->user_points[i+1];
	}
	curve_update_part(index, c);
}

// move a point to another position
int curve_move_point(int index, int x, int y, Curve *c)
{
	int new_index = index;
	// fix order
	while (new_index+1 < c->count && c->user_points[new_index+1].x < x){
		c->user_points[new_index] = c->user_points[new_index+1];
		new_index ++;
	}
	while (new_index > 0 && c->user_points[new_index-1].x > x){
		c->user_points[new_index] = c->user_points[new_index-1];
		new_index --;
	}
	// change position
	c->user_points[new_index].x = x;
	c->user_points[new_index].y = y;
	// update curve around old index
	// it is _usually_ not needed to continue to new_index (and nobody ever cares)
	curve_update_part(index, c);
	return new_index;
}

// GTK event handler for the graph widget
static gint graph_events (GtkWidget *widget, GdkEvent *event, PluginData *pd)
{
  static GdkCursorType cursor_type = GDK_TOP_LEFT_ARROW;
  int tx, ty, index, dist;

  gdk_window_get_pointer (pd->graph->window, &tx, &ty, NULL);
	
	if (event->type == GDK_EXPOSE)
	{
		if (pd->graph_pixmap == NULL)
			pd->graph_pixmap = gdk_pixmap_new (pd->graph->window, GRAPH_WIDTH, GRAPH_HEIGHT, -1);
		graph_redraw (pd);
	}	
	
	else if (event->type == GDK_BUTTON_PRESS)
	{
		// Button press: add or grab a point
		index = gdkpoint_bisect(tx, pd->curve_user.user_points, pd->curve_user.count);
		if (index < pd->curve_user.count){
			dist = pd->curve_user.user_points[index].x - tx;
		}
		if (index > 0 && tx - pd->curve_user.user_points[index-1].x < dist) {
			index -= 1;
			dist = tx - pd->curve_user.user_points[index].x;
		}
		if (dist <= GRAPH_HOTSPOT || pd->curve_user.count == USER_POINT_COUNT){
			pd->point_grabbed = curve_move_point(index, tx, ty, &pd->curve_user);
		}
		else {
			pd->point_grabbed = curve_add_point(tx, ty, &pd->curve_user);
		}
		pd->curve_user_changed = TRUE;
		graph_redraw (pd);
		gimp_preview_invalidate(GIMP_PREVIEW(pd->preview));
	}
		
	else if (event->type == GDK_BUTTON_RELEASE)
	{
		// Button release: move a point and remove it if requested
		if (pd->point_grabbed >= 0) {
			if (tx < 0 && pd->point_grabbed > 0) { // if point is not first, remove it
				curve_remove_point(pd->point_grabbed, &pd->curve_user);
			}
			else if (tx >= GRAPH_WIDTH && pd->point_grabbed+1 < pd->curve_user.count){ // if point is not last, remove it
				curve_remove_point(pd->point_grabbed, &pd->curve_user);
			}
			else {
				curve_move_point(pd->point_grabbed, CLAMPED(tx, 0,GRAPH_WIDTH-1), CLAMPED(ty, 0,GRAPH_HEIGHT-1), &pd->curve_user);
			}
			pd->point_grabbed = -1;
			pd->curve_user_changed = TRUE;
			graph_redraw (pd);
			if (pd->do_preview_hd){
				preview_hd(NULL, pd);
			}
			gimp_preview_invalidate(GIMP_PREVIEW(pd->preview));
		}
	}
		
	else if (event->type == GDK_MOTION_NOTIFY)
	{
		// Mouse move: move a previously grabbed point
		if (pd->point_grabbed >= 0){
			pd->point_grabbed = curve_move_point(pd->point_grabbed, CLAMPED(tx, 0,GRAPH_WIDTH-1), CLAMPED(ty, 0,GRAPH_HEIGHT-1), &pd->curve_user);
			pd->curve_user_changed = TRUE;
		  graph_redraw (pd);
			gimp_preview_invalidate(GIMP_PREVIEW(pd->preview));
		}
	}
	
	return FALSE;
}

// Combine wavelets with latest FFT result and show output in the preview widget
void wavelet_preview(PluginData *pd)
{
	int x, y, w, h;
	gimp_preview_get_position (GIMP_PREVIEW(pd->preview), &x, &y);
	gimp_preview_get_size (GIMP_PREVIEW(pd->preview), &w, &h);
	gimp_pixel_rgn_init (&pd->region, pd->drawable, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height, FALSE, TRUE);
	wavelet_apply(pd, x, y, w, h);
	
	gimp_pixel_rgn_set_rect(&pd->region, pd->img_pixels, x, y, w, h);
	gimp_drawable_preview_draw_region(GIMP_DRAWABLE_PREVIEW(pd->preview), &pd->region);
}

// GTK callback: "HD preview" button clicked
// Make a preview in full quality and show it
gint preview_hd(GtkWidget *preview, PluginData *pd)
{
	fft_apply(pd);
	pd->curve_user_changed = FALSE;
	wavelet_preview(pd);
	return FALSE;
}

// GTK callback: "Always preview HD" checkbox was toggled
// Save its value and make a preview if needed
gint preview_hd_toggled(GtkWidget *checkbox, PluginData *pd)
{
	pd->do_preview_hd = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(checkbox));
	if (pd->do_preview_hd && pd->curve_user_changed) {
		fft_apply(pd);
		pd->curve_user_changed = FALSE;
		wavelet_preview(pd);
	}
	return FALSE;
}

// GTK callback: for any whatever reason, the preview must be redrawed
gint preview_invalidated(PluginData *pd, GtkWidget *preview)
{
	wavelet_preview(pd);
	return FALSE;
}

// Create and handle the plugin's dialog
gboolean dialog(PluginData *pd)
{
	gimp_ui_init (PLUG_IN_BINARY, FALSE);
	GtkWidget *dialog, *main_hbox, *hbox_buttons, *preview, *graph, *vbox, *preview_button, *preview_hd_checkbox;
	dialog = gimp_dialog_new ("Frequency Curves", PLUG_IN_BINARY,
														NULL, (GtkDialogFlags)0,
														gimp_standard_help_func, PLUG_IN_NAME,
														GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
														GTK_STOCK_OK,     GTK_RESPONSE_OK,
														NULL);
	
	main_hbox = gtk_hbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (main_hbox), 12);
	gtk_container_add (GTK_CONTAINER(GTK_DIALOG(dialog)->vbox), main_hbox);
	
	curve_init(&pd->curve_user);
	curve_copy(&pd->curve_user, &pd->curve_fft);
	
	pd->preview = gimp_drawable_preview_new (pd->drawable, 0);
	gtk_box_pack_start (GTK_BOX (main_hbox), pd->preview, TRUE, TRUE, 0);
	gtk_widget_show (pd->preview);
	
	g_signal_connect_swapped (pd->preview, "invalidated", G_CALLBACK (preview_invalidated), pd);
	
	vbox = gtk_vbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (vbox), 12);
	gtk_container_add (GTK_CONTAINER(main_hbox), vbox);
	gtk_widget_show(vbox);
	
	graph = pd->graph = gtk_drawing_area_new();
	pd->graph_pixmap = NULL;
	gtk_widget_set_size_request (graph, GRAPH_WIDTH, GRAPH_HEIGHT);
	gtk_widget_set_events (graph, GDK_EXPOSURE_MASK | GDK_POINTER_MOTION_MASK | 
	                              GDK_POINTER_MOTION_HINT_MASK | GDK_ENTER_NOTIFY_MASK | 
	                              GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | 
	                              GDK_BUTTON1_MOTION_MASK);
	gtk_container_add (GTK_CONTAINER (vbox), graph);
	gtk_widget_show (graph);
	g_signal_connect (graph, "event", G_CALLBACK (graph_events), pd);
	
	hbox_buttons = gtk_hbox_new (FALSE, 12);
	gtk_container_set_border_width (GTK_CONTAINER (hbox_buttons), 12);
	gtk_container_add (GTK_CONTAINER(vbox), hbox_buttons);
	gtk_widget_show(hbox_buttons);
	
	preview_button = gtk_button_new_with_mnemonic ("HD _Preview");
	gtk_box_pack_start (GTK_BOX (hbox_buttons), preview_button, FALSE, FALSE, 0);
	gtk_widget_show (preview_button);
	g_signal_connect (preview_button, "clicked", G_CALLBACK (preview_hd), pd);
	
	preview_hd_checkbox = gtk_check_button_new_with_label("Always preview HD");
	gtk_box_pack_start (GTK_BOX (hbox_buttons), preview_hd_checkbox, FALSE, FALSE, 0);
	gtk_widget_show (preview_hd_checkbox);
	pd->do_preview_hd = FALSE;
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(preview_hd_checkbox), FALSE);
	g_signal_connect (preview_hd_checkbox, "toggled", G_CALLBACK (preview_hd_toggled), pd);
	
	gtk_widget_show(main_hbox);
	gtk_widget_show(dialog);
	fft_prepare(pd);
	histogram_generate(pd);
	wavelet_prepare(pd);
	wavelet_preview(pd);
	gboolean run = (gimp_dialog_run (GIMP_DIALOG (dialog)) == GTK_RESPONSE_OK);
	if (run) {
		// set the region mode to actual writing
		gimp_pixel_rgn_init(&pd->region, pd->drawable, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height, TRUE, TRUE);
		fft_apply(pd);
		// show the result
		gimp_pixel_rgn_set_rect(&pd->region, pd->img_pixels, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height);
		gimp_drawable_flush(pd->drawable);
		gimp_drawable_merge_shadow(pd->drawable->drawable_id, TRUE);
		gimp_drawable_update(pd->drawable->drawable_id, pd->img_offset_x, pd->img_offset_y, pd->img_width, pd->img_height);
		gimp_displays_flush();
	}
	fft_destroy(pd);
	wavelet_destroy(pd);
	gtk_widget_destroy (dialog);
	return run;
}

// GIMP magic
MAIN()

// Register this plugin in GIMP database
void query(void)
{
  // Parameters given to the plugin
  static GimpParamDef args[] = {
    { GIMP_PDB_INT32, (gchar *)"run_mode", (gchar *)"Interactive, non-interactive" },
    { GIMP_PDB_IMAGE, (gchar *)"image", (gchar *)"Input image (unused)" },
    { GIMP_PDB_DRAWABLE, (gchar *)"drawable", (gchar *)"Input drawable" }
  };

  gimp_install_procedure(
    PLUG_IN_NAME,
    "Apply a custom convolution to the whole image",
    "Apply a custom convolution to the whole image",
    PLUG_IN_AUTHOR,
    PLUG_IN_AUTHOR,
    PLUG_IN_VERSION,
    "Frequency Curves...",
    "RGB*, GRAY*",
    GIMP_PLUGIN,
    G_N_ELEMENTS (args), 0,
    args, NULL);
  gimp_plugin_menu_register (PLUG_IN_NAME, "<Image>/Filters/Enhance");
  
}

// Run the plugin
static void
run (const gchar      *name,
     gint              nparams,
     const GimpParam  *param,
     gint             *nreturn_vals,
     GimpParam       **return_vals)
{
	// Return values
  static GimpParam values[1];

  GimpDrawable      *drawable;
  gint sel_x1, sel_y1, sel_x2, sel_y2, w, h, padding;
	PluginData         pd;
  GimpRunMode        run_mode;
  GimpPDBStatusType  status = GIMP_PDB_SUCCESS;

  *nreturn_vals = 1;
  *return_vals  = values;

  if (param[0].type!= GIMP_PDB_INT32) 
	  status=GIMP_PDB_CALLING_ERROR;
  if (param[2].type!=GIMP_PDB_DRAWABLE)
	  status=GIMP_PDB_CALLING_ERROR;

  run_mode = (GimpRunMode) param[0].data.d_int32;
	
	drawable = pd.drawable = gimp_drawable_get(param[2].data.d_drawable);
	gimp_drawable_mask_bounds(drawable->drawable_id, &sel_x1, &sel_y1, &sel_x2, &sel_y2);
	pd.img_width = sel_x2 - sel_x1;
  pd.img_height = sel_y2 - sel_y1;
  pd.img_offset_x = sel_x1;
  pd.img_offset_y = sel_y1;
  pd.img_bpp = gimp_drawable_bpp(drawable->drawable_id);

  pd.point_grabbed = -1;

	if (run_mode == GIMP_RUN_INTERACTIVE)
	{
		dialog(&pd);
		gimp_set_data (PLUG_IN_BINARY, pd.curve_user.user_points, sizeof (GdkPoint) * pd.curve_user.count);
	}
	else if (run_mode == GIMP_RUN_WITH_LAST_VALS)
	{
		fft_prepare(&pd);
		gimp_get_data(PLUG_IN_BINARY, pd.curve_user.user_points);
		pd.curve_user.count = gimp_get_data_size(PLUG_IN_BINARY) / sizeof (GdkPoint);
		gimp_pixel_rgn_init(&pd.region, pd.drawable, pd.img_offset_x, pd.img_offset_y, pd.img_width, pd.img_height, TRUE, TRUE);
		fft_apply(&pd);
		gimp_pixel_rgn_set_rect(&pd.region, pd.img_pixels, pd.img_offset_x, pd.img_offset_y, pd.img_width, pd.img_height);
		gimp_drawable_flush(pd.drawable);
		gimp_drawable_merge_shadow(pd.drawable->drawable_id, TRUE);
		gimp_drawable_update(pd.drawable->drawable_id, pd.img_offset_x, pd.img_offset_y, pd.img_width, pd.img_height);
		fft_destroy(&pd);
		gimp_displays_flush();
	}
	else
	{
	  status = GIMP_PDB_CALLING_ERROR;	
	}
  values[0].type = GIMP_PDB_STATUS;
  values[0].data.d_status = status;
  gimp_drawable_detach(drawable);
}
