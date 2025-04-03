#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x2c635209, "module_layout" },
	{ 0xe59eeb0f, "simple_attr_release" },
	{ 0x3638eb67, "simple_attr_write" },
	{ 0xdfc9b004, "simple_attr_read" },
	{ 0xe75b032b, "generic_file_llseek" },
	{ 0xa555f7ee, "debugfs_remove" },
	{ 0x37a0cba, "kfree" },
	{ 0xf8f7f9b4, "debugfs_create_file" },
	{ 0x7005463f, "debugfs_create_dir" },
	{ 0xa74ccb9b, "pci_enable_device" },
	{ 0xb59830a0, "pci_dev_get" },
	{ 0x9b565c56, "pci_dev_put" },
	{ 0xaf88e69b, "kmem_cache_alloc_trace" },
	{ 0x30a93ed, "kmalloc_caches" },
	{ 0x33b5fc4b, "pci_get_device" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0xead99a33, "pci_read_config_dword" },
	{ 0x92997ed8, "_printk" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xcf4d5c6a, "simple_attr_open" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "B1215F6EC7010DD74EAD967");
