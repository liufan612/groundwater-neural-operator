"""Read cellwise MF6 transport budgets without replacing missing data with zeros."""
import numpy as np
import flopy

def transport_budget_diagnostics(budget_path,grid_path):
    grid=flopy.mf6.utils.MfGrdFile(str(grid_path))
    ia,ja=grid.ia,grid.ja
    n=len(ia)-1
    source_nodes=np.repeat(np.arange(n),np.diff(ia))
    off_diagonal=source_nodes!=ja
    unique_edges=source_nodes<ja
    layer_size=int(grid.nrow)*int(grid.ncol)
    vertical=unique_edges&(source_nodes//layer_size!=ja//layer_size)
    rows=[]
    with flopy.utils.CellBudgetFile(str(budget_path),precision='double') as reader:
        labels={s.strip() for s in reader.get_unique_record_names(decode=True)}
        required={'FLOW-JA-FACE','STORAGE-AQUEOUS','SOURCE-SINK MIX'}
        if not required<=labels:raise ValueError(f'Missing cell budget records: {required-labels}')
        for time,pair in zip(reader.get_times(),reader.get_kstpkper()):
            q=reader.get_data(text='FLOW-JA-FACE',kstpkper=pair)[0].ravel()
            source=np.zeros(n)
            for block in reader.get_data(text='SOURCE-SINK MIX',kstpkper=pair):
                np.add.at(source,block['node']-1,block['q'])
            stored=reader.get_data(text='STORAGE-AQUEOUS',kstpkper=pair)[0].ravel()
            outward=np.zeros(n)
            # MF6 off-diagonal FLOW-JA-FACE is positive INTO the row cell.
            np.add.at(outward,source_nodes[off_diagonal],-q[off_diagonal])
            residual=-stored+outward-source
            scale=max(np.abs(source).max(),np.abs(stored).max(),1e-12)
            rows.append(dict(time_days=float(time),max_cell_residual=float(abs(residual).max()),
                max_cell_relative_residual=float(abs(residual).max()/scale),
                global_internal_cancellation=float(outward.sum()),
                global_mass_residual=float(residual.sum()),
                absolute_vertical_mass_exchange=float(abs(q[vertical]).sum())))
    return rows
