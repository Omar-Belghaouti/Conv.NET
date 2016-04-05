__kernel void Im2col ( 	__read_only __global float* A,
						__write_only __global float* B,
						int A_width,
						int B_width,
						int filter_size,
						int n_receptiveFields
                               )
{

    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

	// move to the beginning of channel that we are working on
	uint i_channel_beginning = A_width * A_width * (i / (filter_size * filter_size) );
	
	// move to position corresponding to current receptive field j
	uint i_receptive_field = (j % B_width) + A_width * (j / B_width);
	
	// move to the input value corresponding to filter element i 
	uint i_filter_element = (i % filter_size) + A_width * ( (i % (filter_size * filter_size)) / filter_size );
    
	
	B[i*n_receptiveFields+j]= A[ i_channel_beginning + i_receptive_field + i_filter_element ];
}