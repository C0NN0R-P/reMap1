// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/bitops.h>

#define MODULE_NAME "skx_decoder"

#define GET_BITFIELD(v, lo, hi) (((v) & GENMASK_ULL((hi), (lo))) >> (lo))
#define BIT_ULL(nr) (1ULL << (nr))

#define NUM_IMC 2
#define NUM_CHANNELS 3
#define NUM_DIMMS 2
#define SKX_MAX_TAD 8
#define SKX_MAX_RIR 4

struct skx_dimm {
	u8 rowbits, colbits;
};

struct skx_channel {
	struct pci_dev *cdev;
	struct skx_dimm dimms[NUM_DIMMS];
};

struct skx_imc {
	u8 mc, lmc, src_id, node_id;
	struct skx_channel chan[NUM_CHANNELS];
};

struct skx_dev {
	struct list_head list;
	u8 bus[4];
	struct pci_dev *sad_all;
	u32 mcroute;
	struct skx_imc imc[NUM_IMC];
};

static LIST_HEAD(skx_edac_list);
static int skx_num_sockets;

#define SKX_GET_SAD(d, i, reg) pci_read_config_dword((d)->sad_all, 0x60 + 8 * (i), &reg)
#define SKX_SAD_LIMIT(sad) (((u64)GET_BITFIELD((sad), 7, 26) << 26) | 0x3FFFFFF)
#define SKX_SAD_ENABLE(sad) GET_BITFIELD((sad), 0, 0)
#define SKX_GET_TADBASE(d, mc, i, reg) pci_read_config_dword((d)->imc[mc].chan[0].cdev, 0x850 + 4 * (i), &reg)
#define SKX_GET_TADWAYNESS(d, mc, i, reg) pci_read_config_dword((d)->imc[mc].chan[0].cdev, 0x880 + 4 * (i), &reg)
#define SKX_TAD_BASE(b) ((u64)GET_BITFIELD((b), 12, 31) << 26)
#define SKX_TAD_LIMIT(b) (((u64)GET_BITFIELD((b), 12, 31) << 26) | 0x3FFFFFF)
#define SKX_TAD_CHNWAYS(b) (GET_BITFIELD((b), 8, 9) + 1)
#define SKX_TAD_SKTWAYS(b) (1 << GET_BITFIELD((b), 10, 11))
#define SKX_GET_RIRWAYNESS(d, mc, ch, i, reg) \
	pci_read_config_dword((d)->imc[mc].chan[ch].cdev, 0x108 + 4 * (i), &reg)
#define SKX_RIR_VALID(b) GET_BITFIELD((b), 31, 31)
#define SKX_RIR_LIMIT(b) (((u64)GET_BITFIELD((b), 1, 11) << 29) | 0x1FFFFFFF)
#define SKX_RIR_WAYS(b) (1 << GET_BITFIELD((b), 28, 29))

static struct dentry *skx_debug_dir;
static u64 phys_addr;

// Find socket based on PCI bus ID (from 0x2016 devices)
static struct skx_dev *get_skx_dev(u8 bus, u8 idx)
{
	struct skx_dev *d;
	list_for_each_entry(d, &skx_edac_list, list) {
		if (d->bus[idx] == bus)
			return d;
	}
	return NULL;
}

static int get_all_bus_mappings(void)
{
	struct pci_dev *pdev, *prev = NULL;
	struct skx_dev *d;
	u32 reg;
	int ndev = 0;

	while ((pdev = pci_get_device(PCI_VENDOR_ID_INTEL, 0x2016, prev))) {
		ndev++;
		d = kzalloc(sizeof(*d), GFP_KERNEL);
		if (!d) {
			pci_dev_put(pdev);
			return -ENOMEM;
		}
		pci_read_config_dword(pdev, 0xCC, &reg);
		d->bus[0] = GET_BITFIELD(reg, 0, 7);
		d->bus[1] = GET_BITFIELD(reg, 8, 15);
		d->bus[2] = GET_BITFIELD(reg, 16, 23);
		d->bus[3] = GET_BITFIELD(reg, 24, 31);
		list_add_tail(&d->list, &skx_edac_list);
		skx_num_sockets++;
		prev = pdev;
	}
	return ndev;
}

static int get_sad_devices(void)
{
	struct pci_dev *pdev, *prev = NULL;
	struct skx_dev *d;

	while ((pdev = pci_get_device(PCI_VENDOR_ID_INTEL, 0x208e, prev))) {
		d = get_skx_dev(pdev->bus->number, 1);
		if (!d)
			goto fail;
		if (pci_enable_device(pdev)) {
			pr_err(MODULE_NAME ": failed to enable SAD PCI device\n");
			goto fail;
		}
		d->sad_all = pdev;
		pci_dev_get(pdev);
		prev = pdev;
	}
	return 0;
fail:
	pci_dev_put(pdev);
	return -ENODEV;
}

// Real skx_init: discover socket buses and SAD devices
static int skx_init(void)
{
	int rc;

	rc = get_all_bus_mappings();
	if (rc < 0)
		return rc;
	if (rc == 0) {
		pr_info(MODULE_NAME ": no 0x2016 mapping devices found\n");
		return -ENODEV;
	}

	rc = get_sad_devices();
	if (rc < 0) {
		pr_err(MODULE_NAME ": SAD device probing failed\n");
		return rc;
	}

	pr_info(MODULE_NAME ": discovered %d sockets\n", skx_num_sockets);
	return 0;
}

static void skx_exit(void)
{
	struct skx_dev *d, *tmp;
	list_for_each_entry_safe(d, tmp, &skx_edac_list, list) {
		if (d->sad_all)
			pci_dev_put(d->sad_all);
		list_del(&d->list);
		kfree(d);
	}
	pr_info(MODULE_NAME ": cleaned up\n");
}


static int debugfs_u64_set(void *data, u64 val)
{
    u64 addr = val;
    u32 sad, base, wayness, rirway;
    u64 limit, prev_limit = 0;
    int i, tad_index, rir_index, mc, ch;
    struct skx_dev *d;

    pr_info(MODULE_NAME ": received address 0x%llx", addr);

    list_for_each_entry(d, &skx_edac_list, list) {
        for (i = 0; i < 24; i++) {
            if (!d->sad_all) continue;
            SKX_GET_SAD(d, i, sad);
            if (!SKX_SAD_ENABLE(sad)) continue;
            limit = SKX_SAD_LIMIT(sad);
            if (addr >= prev_limit && addr <= limit) {
                pr_info(MODULE_NAME ": matched SAD %d (limit 0x%llx)", i, limit);
                for (mc = 0; mc < NUM_IMC; mc++) {
                    if (!d->imc[mc].chan[0].cdev) continue;
                    for (tad_index = 0; tad_index < SKX_MAX_TAD; tad_index++) {
                        SKX_GET_TADBASE(d, mc, tad_index, base);
                        SKX_GET_TADWAYNESS(d, mc, tad_index, wayness);
                        if (addr >= SKX_TAD_BASE(base) && addr <= SKX_TAD_LIMIT(wayness)) {
                            int chanways = SKX_TAD_CHNWAYS(wayness);
                            int sktways = SKX_TAD_SKTWAYS(wayness);
                            pr_info(MODULE_NAME ": IMC %d matched TAD %d | chnways=%d sktways=%d",
                                    mc, tad_index, chanways, sktways);
                            for (ch = 0; ch < NUM_CHANNELS; ch++) {
                                if (!d->imc[mc].chan[ch].cdev) continue;
                                for (rir_index = 0; rir_index < SKX_MAX_RIR; rir_index++) {
                                    SKX_GET_RIRWAYNESS(d, mc, ch, rir_index, rirway);
                                    if (!SKX_RIR_VALID(rirway)) continue;
                                    if (addr <= SKX_RIR_LIMIT(rirway)) {
                                        int ways = SKX_RIR_WAYS(rirway);
                                        if (ways == 0) {
                                            pr_warn(MODULE_NAME ": RIR %d has 0 ways – skipping", rir_index);
                                            continue;
                                        }
                                        u64 rank_addr = addr / ways;
                                        u64 row = (rank_addr >> 18) & 0x3FFF;
                                        u64 col = (rank_addr >> 3) & 0x1FFF;
                                        u64 bank = (rank_addr >> 15) & 0x7;
                                        pr_info(MODULE_NAME ": Channel %d matched RIR %d | Rank addr 0x%llx | ways=%d",
                                                ch, rir_index, rank_addr, ways);
                                        pr_info(MODULE_NAME ": ROW: 0x%llx COL: 0x%llx BANK: %llu", row, col, bank);
                                        return 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            prev_limit = limit + 1;
        }
    }

    pr_info(MODULE_NAME ": address 0x%llx did not match any SAD/TAD/RIR", addr);
    return 0;
}


DEFINE_SIMPLE_ATTRIBUTE(fops_u64_wo, NULL, debugfs_u64_set, "%llu");

static int __init skx_decoder_init(void)
{
	int rc = skx_init();
	if (rc < 0)
		return rc;

	skx_debug_dir = debugfs_create_dir("skx_decode", NULL);
	if (!skx_debug_dir)
		return -ENOMEM;

	debugfs_create_file("addr", 0200, skx_debug_dir, &phys_addr, &fops_u64_wo);
	return 0;
}

static void __exit skx_decoder_exit(void)
{
	skx_exit();
	debugfs_remove_recursive(skx_debug_dir);
	pr_info(MODULE_NAME ": exiting\n");
}

module_init(skx_decoder_init);
module_exit(skx_decoder_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Connor Pfreundschuh");
MODULE_DESCRIPTION("SKX Address Decoder");
