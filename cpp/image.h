#ifndef IMAGE_H_
#define IMAGE_H_

// minimalistic (3D) image class
template <class T>
class image {
    public:
        image(const int nx, const int ny, const int nz = 1): nx_(nx), ny_(ny), nz_(nz) { data_ = new T[nx_*ny_*nz_]; }
        ~image() { delete [] data_; }
        int nx() const { return nx_; }
        int ny() const { return ny_; }
        int nz() const { return nz_; }
        int size() const { return nx_*ny_*nz_; }
        T get(const int i) const { return data_[i]; }
        T get(const int x, const int y, const int z=0) const { return data_[(z*ny_+y)*nx_+x]; }
        void set(const int i, const T val) { data_[i] = val; }
        void set(const int x, const int y, const T val) { data_[y*nx_+x] = val; }
        void set(const int x, const int y, const int z, const T val) { data_[(z*ny_+y)*nx_+x] = val; }
    private:
        int nx_, ny_, nz_;
        T *data_;
};

#endif /* IMAGE_H_ */
