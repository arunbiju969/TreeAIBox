"""Tree centroid computation and caching. (TreeVisualizerGUI mixin)."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.spatial import KDTree


class CentroidsMixin:
    """Tree centroid computation and caching."""

    def calculate_trees_centroid(self, tree_ids):
        """Calculate the centroid of multiple trees combined."""
        all_points = []
        itc_values = np.array(self.las_data['itc'])

        for tree_id in tree_ids:
            mask = itc_values == tree_id

            # Apply tree-point filter if enabled
            tree_mask = self.get_tree_point_mask()
            if tree_mask is not None:
                mask = mask & tree_mask

            if np.any(mask):
                points = np.column_stack([
                    np.array(self.las_data.x)[mask],
                    np.array(self.las_data.y)[mask],
                    np.array(self.las_data.z)[mask]
                ])
                all_points.append(points)

        if not all_points:
            return None

        # Calculate centroid of all points combined
        merged_points = np.vstack(all_points)
        centroid = np.mean(merged_points, axis=0)
        return centroid[:2]  # Return only X, Y coordinates


    def calculate_tree_centroid(self, tree_id):
        """Calculate the centroid of a single tree (now uses cache if available)."""
        # Check cache first
        if tree_id in self.tree_centroids_cache:
            return self.tree_centroids_cache[tree_id]
            
        # Fallback to original calculation
        itc_values = np.array(self.las_data['itc'])
        mask = itc_values == tree_id

        # Apply tree-point filter if enabled
        tree_mask = self.get_tree_point_mask()
        if tree_mask is not None:
            mask = mask & tree_mask

        if not np.any(mask):
            return None

        points = np.column_stack([
            np.array(self.las_data.x)[mask],
            np.array(self.las_data.y)[mask],
            np.array(self.las_data.z)[mask]
        ])

        centroid = np.mean(points, axis=0)
        return centroid[:2]  # Return only X, Y coordinates


    def compute_single_centroid(self, tree_id, itc_values, x_coords, y_coords, z_coords, filter_mask=None):
        """Compute centroid for a single tree (used for parallel processing)."""
        mask = itc_values == tree_id
        if filter_mask is not None:
            mask = mask & filter_mask
            
        if not np.any(mask):
            return tree_id, None
            
        points = np.column_stack([x_coords[mask], y_coords[mask], z_coords[mask]])
        centroid = np.mean(points, axis=0)
        return tree_id, centroid[:2]


    def precompute_tree_centroids(self):
        """Pre-compute centroids for all trees to speed up neighbor calculations."""
        if self.las_data is None or len(self.unique_itc_values) == 0:
            return
            
        self.tree_centroids_cache = {}
        itc_values = np.array(self.las_data['itc'])
        x_coords = np.array(self.las_data.x)
        y_coords = np.array(self.las_data.y)
        z_coords = np.array(self.las_data.z)
        
        # Apply tree-point filter if enabled
        filter_mask = self.get_tree_point_mask()
        
        # Use parallel processing for centroid computation
        num_workers = min(multiprocessing.cpu_count(), len(self.unique_itc_values))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all centroid computation tasks
            future_to_tree = {
                executor.submit(self.compute_single_centroid, tree_id, itc_values, x_coords, y_coords, z_coords, filter_mask): tree_id 
                for tree_id in self.unique_itc_values
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tree):
                tree_id = future_to_tree[future]
                try:
                    tid, centroid = future.result()
                    if centroid is not None:
                        self.tree_centroids_cache[tid] = centroid
                except Exception as exc:
                    print(f'Tree {tree_id} generated an exception: {exc}')
        
        # Build KDTree for fast spatial queries
        if self.tree_centroids_cache:
            centroids_array = np.array(list(self.tree_centroids_cache.values()))
            self.centroids_kdtree = KDTree(centroids_array)
            # Record the tree IDs in the SAME row order the KDTree was built
            # from, so query indices can be mapped back to tree IDs correctly.
            self.centroids_kdtree_tree_ids = list(self.tree_centroids_cache.keys())
            print(f"Built KDTree with {len(centroids_array)} centroids")
        
        print(f"Pre-computed centroids for {len(self.tree_centroids_cache)} trees using {num_workers} parallel workers")


    def on_filter_changed(self):
        """Clear centroids cache when the tree-point filter changes and refresh the view."""
        if self.las_data is not None and len(self.unique_itc_values) > 0:
            self.tree_centroids_cache = {}
            self.centroids_kdtree = None
            self.centroids_kdtree_tree_ids = None
            print("Cleared centroids cache due to filter change (will be recomputed on next neighbor search)")
        # Reflect the filter change in the current 3D view immediately
        if getattr(self, 'current_tree_points', None) is not None:
            self.visualize_tree()


